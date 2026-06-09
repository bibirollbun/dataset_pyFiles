import os
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import logging
from pydicom.pixel_data_handlers.util import apply_voi_lut
import seaborn as sns
from scipy import stats


# 配置日志记录
logging.basicConfig(
    filename='dicom_processing.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


print("===== 数据描述性分析 =====")
# 加载训练标签
print("加载数据...")
train_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
localizers_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')

# 基本数据信息
print(f"训练数据形状: {train_df.shape}")
print(f"定位器数据形状: {localizers_df.shape}")

# 显示数据基本信息
print("\n训练数据基本信息:")
print(train_df.info())



print("\n训练数据描述性统计:")
print(train_df.describe())


# 定义动脉瘤位置标签
location_labels = [
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
    'Other Posterior Circulation'
]

all_label_cols = ['Aneurysm Present'] + location_labels


# 检查标签分布
print("\n===== 标签分布分析 =====")
print("\n主要标签 - 动脉瘤存在分布:")
print(train_df['Aneurysm Present'].value_counts())
print(train_df['Aneurysm Present'].value_counts(normalize=True))

print("\n各位置动脉瘤分布:")
for col in location_labels:
    if col in train_df.columns:
        count = train_df[col].value_counts()
        print(f"\n{col}:")
        print(f"存在: {count.get(1, 0)}, 不存在: {count.get(0, 0)}")
        if 1 in count:
            print(f"比例: {count[1]/len(train_df):.4f}")


# 人口统计学分析
print("\n===== 人口统计学分析 =====")
print("性别分布:")
print(train_df['PatientSex'].value_counts())

print("\n年龄分布:")
print(f"最小值: {train_df['PatientAge'].min()}")
print(f"最大值: {train_df['PatientAge'].max()}")
print(f"平均值: {train_df['PatientAge'].mean():.2f}")
print(f"中位数: {train_df['PatientAge'].median()}")

print("\n成像模式分布:")
print(train_df['Modality'].value_counts())


# 检查缺失值
print("缺失值统计:")
missing_values = train_df.isnull().sum()
print(missing_values[missing_values > 0])


# 处理缺失值
print(f"\n预处理前缺失值总数: {train_df.isnull().sum().sum()}")
train_df.fillna(0, inplace=True)
localizers_df.dropna(inplace=True)
print(f"预处理后缺失值总数: {train_df.isnull().sum().sum()}")


# 异常值处理 - 基于3σ原则，将异常值替换为列均值
print("\n异常值处理:")
numeric_cols = ['PatientAge'] + all_label_cols
for col in numeric_cols:
    if col in train_df.columns:
        z_scores = np.abs(stats.zscore(train_df[col]))
        outliers = z_scores > 3
        if outliers.any():
            mean_val = train_df[col].mean()
            train_df.loc[outliers, col] = mean_val
            print(f"处理 {col} 列的 {outliers.sum()} 个异常值，替换为该列均值 {mean_val:.4f}")


# 重复数据检查
print(f"\n重复数据: {train_df.duplicated().sum()} 行")


# 处理标签列数据类型
for col in all_label_cols:
    if col in train_df.columns:
        if train_df[col].dtype == 'object':
            # 转换文本标签为数值
            train_df[col] = train_df[col].apply(lambda x: 
                1.0 if str(x).strip().lower() in ['true', 'yes', 'present', '1'] else
                0.0 if str(x).strip().lower() in ['false', 'no', 'absent', '0'] else
                float(x) if pd.notna(x) and str(x).strip() else 0.0
            )
        train_df[col] = train_df[col].astype(np.float32)


# 可视化标签分布
plt.figure(figsize=(9, 6))
# 动脉瘤存在分布
plt.subplot(2, 2, 1)
train_df['Aneurysm Present'].value_counts().plot(kind='bar')
plt.title('Distribution of aneurysms')#动脉瘤存在分布
plt.xlabel('Is there an aneurysm')#是否存在动脉瘤
plt.ylabel('Counting')#数量

# 性别分布
plt.subplot(2, 2, 2)
train_df['PatientSex'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')#性别分布
plt.xlabel('Gender')
plt.ylabel('Counting')

# 年龄分布
plt.subplot(2, 2, 3)
plt.hist(train_df['PatientAge'].dropna(), bins=30, edgecolor='black')
plt.title('Age distribution')
plt.xlabel('age')
plt.ylabel('Frequency')

# 成像模式分布
plt.subplot(2, 2, 4)
train_df['Modality'].value_counts().plot(kind='bar')
plt.title('Imaging Mode Distribution')#成像模式分布
plt.xlabel('mode')
plt.ylabel('Counting')

plt.tight_layout()
plt.savefig('data_distribution.png', dpi=300)
plt.show()


# 特征与标签关系分析
print("\n特征与标签关系分析:")

# 性别与动脉瘤存在的关系
if 'PatientSex' in train_df.columns and 'Aneurysm Present' in train_df.columns:
    sex_aneurysm = pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'])
    print("\n性别与动脉瘤存在的关系:")
    print(sex_aneurysm)
    
    # 可视化
    sex_aneurysm.plot(kind='bar', figsize=(10, 6))
    plt.title('The Relationship between Gender and the Presence of Aneurysm')#性别与动脉瘤存在的关系
    plt.xlabel('Gender')
    plt.ylabel('Counting')
    plt.legend(['Aneurysm-free', 'There is an aneurysm.'])
    plt.savefig('sex_vs_aneurysm.png', dpi=300)
    plt.show()


# 年龄与动脉瘤存在的关系
if 'PatientAge' in train_df.columns and 'Aneurysm Present' in train_df.columns:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    train_df[train_df['Aneurysm Present'] == 0]['PatientAge'].hist(alpha=0.7, label='No aneurysm', bins=30)
    train_df[train_df['Aneurysm Present'] == 1]['PatientAge'].hist(alpha=0.7, label='with aneurysm', bins=30)
    plt.title('Age Distribution - by Presence of Aneurysm')#年龄分布 - 按动脉瘤存在
    plt.xlabel('age')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 2, 1)
    age_groups = pd.cut(train_df['PatientAge'], bins=5)
    pd.crosstab(age_groups, train_df['Aneurysm Present']).plot(kind='bar', figsize=(12, 6))
    plt.title('The relationship between age groups and aneurysm presence')#年龄组与动脉瘤存在的关系
    plt.xlabel('age group')#年龄组
    plt.ylabel('Counting')
    plt.legend(['Aneurysm-free', 'There is an aneurysm'])
    plt.tight_layout()
    plt.savefig('age_vs_aneurysm.png', dpi=300)
    plt.show()


# 成像模式与动脉瘤存在的关系
if 'Modality' in train_df.columns and 'Aneurysm Present' in train_df.columns:
    modality_aneurysm = pd.crosstab(train_df['Modality'], train_df['Aneurysm Present'])
    print("\n成像模式与动脉瘤存在的关系:")
    print(modality_aneurysm)
    
    modality_aneurysm.plot(kind='bar', figsize=(10, 6))
    plt.title('The relationship between imaging modes and aneurysm presence')#成像模式与动脉瘤存在的关系
    plt.xlabel('Imaging mode')#成像模式
    plt.ylabel('Counting')
    plt.legend(['Aneurysm-free', 'There is an aneurysm'])
    plt.savefig('modality_vs_aneurysm.png', dpi=300)
    plt.show()


# 各位置动脉瘤的相关性分析
print("\n各位置动脉瘤的相关性矩阵:")
location_corr = train_df[location_labels].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(location_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Aneurysm-related heat map at various positions')#各位置动脉瘤相关性热图
plt.tight_layout()
plt.savefig('location_correlation.png', dpi=300)
plt.show()


print("\n===== 探索性分析 =====")

# 可视化标签分布
plt.figure(figsize=(9, 6))

# 动脉瘤存在分布
plt.subplot(2, 2, 1)
train_df['Aneurysm Present'].value_counts().plot(kind='bar')
plt.title('Distribution of aneurysms')#动脉瘤存在分布
plt.xlabel('Is there an aneurysm')#是否存在动脉瘤
plt.ylabel('Counting')#数量

# 性别分布
plt.subplot(2, 2, 2)
train_df['PatientSex'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')#性别分布
plt.xlabel('Gender')
plt.ylabel('Counting')

# 年龄分布
plt.subplot(2, 2, 3)
plt.hist(train_df['PatientAge'].dropna(), bins=30, edgecolor='black')
plt.title('Age distribution')
plt.xlabel('age')
plt.ylabel('Frequency')

# 成像模式分布
plt.subplot(2, 2, 4)
train_df['Modality'].value_counts().plot(kind='bar')
plt.title('Imaging Mode Distribution')#成像模式分布
plt.xlabel('mode')
plt.ylabel('Counting')

plt.tight_layout()
plt.savefig('data_distribution.png', dpi=300)
plt.show()

# 特征与标签关系分析
print("\n特征与标签关系分析:")

# 性别与动脉瘤存在的关系
if 'PatientSex' in train_df.columns and 'Aneurysm Present' in train_df.columns:
    sex_aneurysm = pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'])
    print("\n性别与动脉瘤存在的关系:")
    print(sex_aneurysm)
    
    # 可视化
    sex_aneurysm.plot(kind='bar', figsize=(10, 6))
    plt.title('The Relationship between Gender and the Presence of Aneurysm')#性别与动脉瘤存在的关系
    plt.xlabel('Gender')
    plt.ylabel('Counting')
    plt.legend(['Aneurysm-free', 'There is an aneurysm.'])
    plt.savefig('sex_vs_aneurysm.png', dpi=300)
    plt.show()

# 年龄与动脉瘤存在的关系
if 'PatientAge' in train_df.columns and 'Aneurysm Present' in train_df.columns:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    train_df[train_df['Aneurysm Present'] == 0]['PatientAge'].hist(alpha=0.7, label='无动脉瘤', bins=30)
    train_df[train_df['Aneurysm Present'] == 1]['PatientAge'].hist(alpha=0.7, label='有动脉瘤', bins=30)
    plt.title('Age Distribution - by Presence of Aneurysm')#年龄分布 - 按动脉瘤存在
    plt.xlabel('age')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    age_groups = pd.cut(train_df['PatientAge'], bins=5)
    pd.crosstab(age_groups, train_df['Aneurysm Present']).plot(kind='bar', figsize=(12, 6))
    plt.title('The relationship between age groups and aneurysm presence')#年龄组与动脉瘤存在的关系
    plt.xlabel('age group')#年龄组
    plt.ylabel('Counting')
    plt.legend(['Aneurysm-free', 'There is an aneurysm'])
    plt.tight_layout()
    plt.savefig('age_vs_aneurysm.png', dpi=300)
    plt.show()

# 成像模式与动脉瘤存在的关系
if 'Modality' in train_df.columns and 'Aneurysm Present' in train_df.columns:
    modality_aneurysm = pd.crosstab(train_df['Modality'], train_df['Aneurysm Present'])
    print("\n成像模式与动脉瘤存在的关系:")
    print(modality_aneurysm)
    
    modality_aneurysm.plot(kind='bar', figsize=(10, 6))
    plt.title('The relationship between imaging modes and aneurysm presence')#成像模式与动脉瘤存在的关系
    plt.xlabel('Imaging mode')#成像模式
    plt.ylabel('Counting')
    plt.legend(['Aneurysm-free', 'There is an aneurysm'])
    plt.savefig('modality_vs_aneurysm.png', dpi=300)
    plt.show()

# 各位置动脉瘤的相关性分析
print("\n各位置动脉瘤的相关性矩阵:")
location_corr = train_df[location_labels].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(location_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Aneurysm-related heat map at various positions')#各位置动脉瘤相关性热图
plt.tight_layout()
plt.savefig('location_correlation.png', dpi=300)
plt.show()


print("\n===== 数据集划分 =====")

# 使用分层抽样确保各类别比例一致
train_df, test_df = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df['Aneurysm Present']
)
train_df, val_df = train_test_split(
    train_df, 
    test_size=0.125,  # 0.8 * 0.125 = 0.1
    random_state=42, 
    stratify=train_df['Aneurysm Present']
)

print(f"训练集大小: {len(train_df)}")
print(f"验证集大小: {len(val_df)}")
print(f"测试集大小: {len(test_df)}")

# 显示各数据集标签分布
print("\n训练集动脉瘤存在分布:")
print(train_df['Aneurysm Present'].value_counts(normalize=True))

print("\n验证集动脉瘤存在分布:")
print(val_df['Aneurysm Present'].value_counts(normalize=True))

print("\n测试集动脉瘤存在分布:")
print(test_df['Aneurysm Present'].value_counts(normalize=True))


# ===================== 数据集类定义 =====================
class AneurysmDataset(Dataset):
    def __init__(self, df, root_dir='/kaggle/input/rsna-intracranial-aneurysm-detection/series', 
                 transform=None, is_test=False, location_labels=None):
        self.df = df.copy()
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.location_labels = location_labels
        self.target_shape = (1, 256, 256)  # (通道数, 高度, 宽度)
        self.label_shape = (14,)
        
        self._validate_and_clean_labels()
        
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"根目录不存在: {self.root_dir}")
            
        valid_indices = self._filter_valid_series()
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"过滤后保留的有效样本数: {len(self.df)}")

    def _validate_and_clean_labels(self):
        if not self.location_labels:
            raise ValueError("location_labels 不能为空")
            
        all_label_cols = ['Aneurysm Present'] + self.location_labels
        missing_cols = [col for col in all_label_cols if col not in self.df.columns]
        if missing_cols:
            raise KeyError(f"缺少标签列: {missing_cols}")
        
        for col in all_label_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col].fillna(0.0, inplace=True)
            self.df[col] = self.df[col].astype(np.float32)

    def _filter_valid_series(self):
        valid_indices = []
        for idx in tqdm(range(len(self.df)), desc="过滤有效样本"):
            series_id = self.df.iloc[idx]['SeriesInstanceUID']
            series_path = os.path.join(self.root_dir, series_id)
            if os.path.exists(series_path) and len(os.listdir(series_path)) > 0:
                valid_indices.append(idx)
            else:
                warn_msg = f"路径不存在或为空 - {series_path}"
                logging.warning(warn_msg)
        return valid_indices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        series_id = self.df.iloc[idx]['SeriesInstanceUID']
        series_path = os.path.join(self.root_dir, series_id)
        
        # 初始化返回值
        image = torch.zeros(self.target_shape, dtype=torch.float32)
        label = torch.zeros(self.label_shape, dtype=torch.float32)

        try:
            # 加载DICOM文件列表
            dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
            if not dicom_files:
                warn_msg = f"未找到DICOM文件 - {series_path}"
                logging.warning(warn_msg)
                return image, label
            
            # 按数字排序并选择中间切片
            dicom_files.sort(key=lambda x: int(x.split('.')[0]))
            mid_idx = len(dicom_files) // 2
            dicom_path = os.path.join(series_path, dicom_files[mid_idx])
            
            # 读取DICOM文件
            dicom = pydicom.dcmread(dicom_path)
            
            # 应用VOI LUT增强对比度
            pixel_array = apply_voi_lut(dicom.pixel_array, dicom)
            
            # 处理3D数据
            if len(pixel_array.shape) == 3:
                slice_idx = pixel_array.shape[0] // 2
                pixel_array = pixel_array[slice_idx, :, :]
                logging.warning(f"{series_id} 是3D数据，已取中间切片")
            elif len(pixel_array.shape) != 2:
                error_msg = f"DICOM图像维度异常: {pixel_array.shape}，series_id: {series_id}"
                logging.error(error_msg)
                return image, label
            
            # 转换为float32并应用 rescale 校正
            pixel_array = pixel_array.astype(np.float32)
            if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
                pixel_array = pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept
            
            # 标准化到0-1范围
            img_min, img_max = pixel_array.min(), pixel_array.max()
            if img_max != img_min:
                pixel_array = (pixel_array - img_min) / (img_max - img_min)
            
            # 转换为张量并添加通道维度
            img_tensor = torch.from_numpy(pixel_array).unsqueeze(0)
            
            # 调整尺寸
            resize_transform = transforms.Resize((self.target_shape[1], self.target_shape[2]))
            img_tensor = resize_transform(img_tensor)
            
            if img_tensor.shape != self.target_shape:
                raise RuntimeError(f"图像形状错误: {img_tensor.shape} 预期 {self.target_shape}")
            
            image = img_tensor
            
        except Exception as e:
            error_msg = f"处理DICOM时出错 {series_path}: {str(e)}"
            logging.error(error_msg)
        
        # 处理标签
        try:
            if not self.is_test:
                present = float(self.df.iloc[idx]['Aneurysm Present'])
                locations = self.df.iloc[idx][self.location_labels].values.astype(np.float32)
                label_np = np.concatenate([[present], locations])
                label = torch.FloatTensor(label_np)
        except Exception as e:
            error_msg = f"处理标签时出错 {series_id}: {str(e)}"
            logging.error(error_msg)
        
        # 应用变换
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                logging.error(f"应用变换时出错: {str(e)}")
        
        return image, label


# ===================== 数据加载器准备 =====================
print("\n===== 准备数据加载器 =====")

# 定义图像变换 - 增加数据增强提高模型泛化能力
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 创建数据集实例
train_dataset = AneurysmDataset(
    train_df, 
    transform=train_transform, 
    location_labels=location_labels
)
val_dataset = AneurysmDataset(
    val_df, 
    transform=val_transform, 
    location_labels=location_labels
)
test_dataset = AneurysmDataset(
    test_df, 
    transform=val_transform, 
    is_test=False,  # 保持为False以便评估
    location_labels=location_labels
)



# 创建数据加载器
batch_size = 16
num_workers = 0 if device.type == 'cpu' else 2
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    pin_memory=True if device.type == 'cuda' else False
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=True if device.type == 'cuda' else False
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=True if device.type == 'cuda' else False
)


# 模型定义
class AneurysmViT(nn.Module):
    def __init__(self, num_labels=14):
        super(AneurysmViT, self).__init__()
        
        self.vit_config = ViTConfig(
            image_size=256,
            patch_size=32,
            num_channels=1,
            num_labels=num_labels,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            dropout=0.3,  # 增加dropout防止过拟合
            attention_probs_dropout_prob=0.3
        )
        
        self.vit = ViTModel(self.vit_config)
        
        # 改进分类头
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )
        
    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return torch.sigmoid(logits)


# 初始化模型
model = AneurysmViT().to(device)
print("\n模型结构:")
print(model)

# 计算类别权重 - 更精确地处理类别不平衡
present_count = train_df['Aneurysm Present'].sum()
total_count = len(train_df)
weight_positive = total_count / (2 * present_count) if present_count > 0 else 1.0
weight_negative = total_count / (2 * (total_count - present_count)) if (total_count - present_count) > 0 else 1.0

print(f"\n类别权重 - 阳性: {weight_positive:.2f}, 阴性: {weight_negative:.2f}")


# 加权损失函数
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, output, target):
        # 对第一个标签（是否存在动脉瘤）应用特殊权重
        weights = torch.ones_like(target, device=device)
        weights[:, 0] = self.pos_weight
        
        bce_loss = nn.BCELoss(reduction='none')(output, target)
        return (bce_loss * weights).mean()

# 优化器和调度器
criterion = WeightedBCELoss(pos_weight=weight_positive)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-4, 
    weight_decay=1e-5
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=5, 
    T_mult=2, 
    eta_min=1e-6
)

# 训练参数
num_epochs = 15
best_val_score = 0.0
early_stop_patience = 5
early_stop_counter = 0

# 记录训练过程指标
train_losses = []
val_losses = []
train_aucs = []
val_aucs = []


# ===================== 训练循环 =====================
print("\n===== 开始训练 =====")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_auc = 0.0
    train_auc_count = 0
    
    # 训练循环带进度条
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 训练")
    for images, labels in train_pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        
        # 计算AUC
        labels_np = labels.cpu().detach().numpy()
        outputs_np = outputs.cpu().detach().numpy()
        
        try:
            if len(np.unique(labels_np[:, 0])) >= 2:
                auc = roc_auc_score(labels_np[:, 0], outputs_np[:, 0])
                train_auc += auc * images.size(0)
                train_auc_count += images.size(0)
        except ValueError as e:
            logging.error(f"训练AUC计算错误: {str(e)}")
        
        # 更新进度条
        train_pbar.set_postfix({"batch_loss": loss.item()})
    
    # 计算平均训练损失和AUC
    train_loss /= len(train_loader.dataset)
    train_auc = train_auc / train_auc_count if train_auc_count > 0 else 0.5
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_scores = []
    val_main_aucs = []
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 验证")
        for images, labels in val_pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            # 计算AUC
            labels_np = labels.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            auc_scores = []
            
            for i in range(labels_np.shape[1]):
                try:
                    if len(np.unique(labels_np[:, i])) >= 2:
                        auc_scores.append(roc_auc_score(labels_np[:, i], outputs_np[:, i]))
                    else:
                        auc_scores.append(0.5)
                except:
                    auc_scores.append(0.5)
            
            main_auc = auc_scores[0]
            other_auc = np.mean(auc_scores[1:])
            val_scores.append(0.5 * (main_auc + other_auc))
            val_main_aucs.append(main_auc)
            
            # 更新进度条
            val_pbar.set_postfix({"batch_loss": loss.item()})
    
    # 计算平均验证损失和分数
    val_loss /= len(val_loader.dataset)
    val_score = np.mean(val_scores)
    val_main_auc = np.mean(val_main_aucs)
    
    # 更新学习率调度器
    scheduler.step()
    
    # 记录指标
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_aucs.append(train_auc)
    val_aucs.append(val_main_auc)
    
    # 保存最佳模型
    if val_score > best_val_score:
        best_val_score = val_score
        torch.save(model.state_dict(), 'best_model.pth')
        early_stop_counter = 0
        print("保存最佳模型!")
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print("早停: 验证分数连续多轮无提升")
            break
    
    # 打印 epoch 结果
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"训练损失: {train_loss:.4f} | 训练主标签AUC: {train_auc:.4f}")
    print(f"验证损失: {val_loss:.4f} | 验证主标签AUC: {val_main_auc:.4f} | 验证综合分数: {val_score:.4f}")
    print("-" * 80)

print("训练完成!")


# ===================== 绘制训练历史 =====================
print("\n绘制训练历史...")
plt.figure(figsize=(14, 6))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Training and Validation Loss Curve')#训练和验证损失曲线
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# AUC曲线
plt.subplot(1, 2, 2)
plt.plot(train_aucs, label='Training AUC')
plt.plot(val_aucs, label='Verification AUC')
plt.title('Training and Validation AUC Curve')#训练和验证AUC曲线
plt.xlabel('Epoch')
plt.ylabel('AUC value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
plt.show()


print("\n===== 测试集评估 =====")
# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

test_scores = []
test_aucs = []
all_auc_scores = []

with torch.no_grad():
    test_pbar = tqdm(test_loader, desc="测试集评估")
    for images, labels in test_pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        # 计算AUC
        labels_np = labels.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
        
        # 计算每个标签的AUC
        auc_scores = []
        for i in range(labels_np.shape[1]):
            try:
                auc = roc_auc_score(labels_np[:, i], outputs_np[:, i])
                auc_scores.append(auc)
            except ValueError:
                auc_scores.append(0.5)
        
        all_auc_scores.append(auc_scores)
        
        # 计算加权分数
        main_auc = auc_scores[0]
        other_auc = np.mean(auc_scores[1:])
        weighted_score = 0.5 * (main_auc + other_auc)
        
        test_scores.append(weighted_score)
        test_aucs.append(main_auc)


# 计算平均分数
test_score = np.mean(test_scores)
test_auc = np.mean(test_aucs)

print(f"\n测试集主标签AUC: {test_auc:.4f}")
print(f"测试集综合分数: {test_score:.4f}")

# 计算每个位置的平均AUC
mean_auc_per_label = np.mean(all_auc_scores, axis=0)
print("\n每个标签的平均AUC:")
for i, label in enumerate(['Aneurysm Present'] + location_labels):
    print(f"{label}: {mean_auc_per_label[i]:.4f}")


# ===================== 可视化预测结果 =====================
print("\n可视化预测结果...")
def visualize_predictions(model, dataset, num_samples=5):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    plt.figure(figsize=(15, 4*num_samples))
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(image_tensor).cpu().numpy()[0]
        
        # 显示图像
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(image[0], cmap='gray')
        plt.title(f"True value: {label[0].item():.0f}, Predicted Value: {pred[0]:.2f}")
        plt.axis('off')
        
        # 显示位置预测
        plt.subplot(num_samples, 2, 2*i+2)
        plt.barh(location_labels, pred[1:])
        plt.xlim(0, 1)
        plt.title("Position Prediction Probability")#位置预测概率
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=300)
    plt.show()

# 可视化测试集预测结果
visualize_predictions(model, test_dataset, num_samples=5)


print("\n生成提交文件...")
def generate_submission(model, test_loader, dataset, device):
    model.eval()
    all_preds = []
    all_ids = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="生成预测结果")
        for images, _ in pbar:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.cpu().numpy()
            all_preds.append(preds)
    
    # 获取真实的SeriesInstanceUID
    all_ids = dataset.df['SeriesInstanceUID'].values
    
    all_preds = np.concatenate(all_preds)
    
    # 创建提交DataFrame
    submission_df = pd.DataFrame(all_preds, columns=['Aneurysm Present'] + location_labels)
    submission_df.insert(0, 'SeriesInstanceUID', all_ids)
    
    return submission_df

# 生成提交文件（保存为parquet格式，替换原csv逻辑）
submission_df = generate_submission(model, test_loader, test_dataset, device)
# 使用snappy压缩减少文件体积，兼容大多数比赛平台
submission_df.to_parquet('submission.parquet', index=False, compression='snappy')

# 验证文件是否生成成功
if os.path.exists('submission.parquet'):
    file_size = os.path.getsize('submission.parquet') / 1024 / 1024  # 转换为MB
    print(f"提交文件已成功保存为 submission.parquet，文件大小: {file_size:.2f} MB")
else:
    print("错误：提交文件未生成")


# import os
# import numpy as np
# import pandas as pd
# import pydicom
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from transformers import ViTModel, ViTConfig
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import warnings
# import logging
# from pydicom.pixel_data_handlers.util import apply_voi_lut
# import seaborn as sns
# from scipy import stats

# # 配置日志记录
# logging.basicConfig(
#     filename='dicom_processing.log',
#     level=logging.WARNING,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# warnings.filterwarnings('ignore')

# # 设置随机种子保证可重复性
# torch.manual_seed(42)
# np.random.seed(42)

# # 检查GPU可用性
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"使用设备: {device}")

# # ===================== 数据描述性分析 =====================
# print("===== 数据描述性分析 =====")

# # 加载训练标签
# print("加载数据...")
# train_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
# localizers_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')

# # 基本数据信息
# print(f"训练数据形状: {train_df.shape}")
# print(f"定位器数据形状: {localizers_df.shape}")

# # 显示数据基本信息
# print("\n训练数据基本信息:")
# print(train_df.info())

# print("\n训练数据描述性统计:")
# print(train_df.describe())

# # 定义动脉瘤位置标签
# location_labels = [
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
#     'Other Posterior Circulation'
# ]

# all_label_cols = ['Aneurysm Present'] + location_labels

# # 检查标签分布
# print("\n===== 标签分布分析 =====")
# print("\n主要标签 - 动脉瘤存在分布:")
# print(train_df['Aneurysm Present'].value_counts())
# print(train_df['Aneurysm Present'].value_counts(normalize=True))

# print("\n各位置动脉瘤分布:")
# for col in location_labels:
#     if col in train_df.columns:
#         count = train_df[col].value_counts()
#         print(f"\n{col}:")
#         print(f"存在: {count.get(1, 0)}, 不存在: {count.get(0, 0)}")
#         if 1 in count:
#             print(f"比例: {count[1]/len(train_df):.4f}")

# # 人口统计学分析
# print("\n===== 人口统计学分析 =====")
# print("性别分布:")
# print(train_df['PatientSex'].value_counts())

# print("\n年龄分布:")
# print(f"最小值: {train_df['PatientAge'].min()}")
# print(f"最大值: {train_df['PatientAge'].max()}")
# print(f"平均值: {train_df['PatientAge'].mean():.2f}")
# print(f"中位数: {train_df['PatientAge'].median()}")

# print("\n成像模式分布:")
# print(train_df['Modality'].value_counts())

# # ===================== 数据清洗 =====================
# print("\n===== 数据清洗 =====")

# # 检查缺失值
# print("缺失值统计:")
# missing_values = train_df.isnull().sum()
# print(missing_values[missing_values > 0])

# # 处理缺失值
# print(f"\n预处理前缺失值总数: {train_df.isnull().sum().sum()}")
# train_df.fillna(0, inplace=True)
# localizers_df.dropna(inplace=True)
# print(f"预处理后缺失值总数: {train_df.isnull().sum().sum()}")

# # 异常值检测 - 使用3σ原则
# print("\n异常值检测:")
# numeric_cols = ['PatientAge'] + all_label_cols
# for col in numeric_cols:
#     if col in train_df.columns:
#         z_scores = np.abs(stats.zscore(train_df[col]))
#         outliers = np.sum(z_scores > 3)
#         print(f"{col}: {outliers} 个异常值")

# # 重复数据检查
# print(f"\n重复数据: {train_df.duplicated().sum()} 行")

# # 处理标签列数据类型
# for col in all_label_cols:
#     if col in train_df.columns:
#         if train_df[col].dtype == 'object':
#             # 转换文本标签为数值
#             train_df[col] = train_df[col].apply(lambda x: 
#                 1.0 if str(x).strip().lower() in ['true', 'yes', 'present', '1'] else
#                 0.0 if str(x).strip().lower() in ['false', 'no', 'absent', '0'] else
#                 float(x) if pd.notna(x) and str(x).strip() else 0.0
#             )
#         train_df[col] = train_df[col].astype(np.float32)

# # ===================== 探索性分析 =====================
# print("\n===== 探索性分析 =====")

# # 可视化标签分布
# plt.figure(figsize=(15, 10))

# # 动脉瘤存在分布
# plt.subplot(2, 2, 1)
# train_df['Aneurysm Present'].value_counts().plot(kind='bar')
# plt.title('Distribution of aneurysms')#动脉瘤存在分布
# plt.xlabel('Is there an aneurysm')#是否存在动脉瘤
# plt.ylabel('Counting')#数量

# # 性别分布
# plt.subplot(2, 2, 2)
# train_df['PatientSex'].value_counts().plot(kind='bar')
# plt.title('Gender Distribution')#性别分布
# plt.xlabel('Gender')
# plt.ylabel('Counting')

# # 年龄分布
# plt.subplot(2, 2, 3)
# plt.hist(train_df['PatientAge'].dropna(), bins=30, edgecolor='black')
# plt.title('Age distribution')
# plt.xlabel('age')
# plt.ylabel('Frequency')

# # 成像模式分布
# plt.subplot(2, 2, 4)
# train_df['Modality'].value_counts().plot(kind='bar')
# plt.title('Imaging Mode Distribution')#成像模式分布
# plt.xlabel('mode')
# plt.ylabel('Counting')

# plt.tight_layout()
# plt.savefig('data_distribution.png', dpi=300)
# plt.show()

# # 特征与标签关系分析
# print("\n特征与标签关系分析:")

# # 性别与动脉瘤存在的关系
# if 'PatientSex' in train_df.columns and 'Aneurysm Present' in train_df.columns:
#     sex_aneurysm = pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'])
#     print("\n性别与动脉瘤存在的关系:")
#     print(sex_aneurysm)
    
#     # 可视化
#     sex_aneurysm.plot(kind='bar', figsize=(10, 6))
#     plt.title('The Relationship between Gender and the Presence of Aneurysm')#性别与动脉瘤存在的关系
#     plt.xlabel('Gender')
#     plt.ylabel('Counting')
#     plt.legend(['Aneurysm-free', 'There is an aneurysm.'])
#     plt.savefig('sex_vs_aneurysm.png', dpi=300)
#     plt.show()

# # 年龄与动脉瘤存在的关系
# if 'PatientAge' in train_df.columns and 'Aneurysm Present' in train_df.columns:
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     train_df[train_df['Aneurysm Present'] == 0]['PatientAge'].hist(alpha=0.7, label='无动脉瘤', bins=30)
#     train_df[train_df['Aneurysm Present'] == 1]['PatientAge'].hist(alpha=0.7, label='有动脉瘤', bins=30)
#     plt.title('Age Distribution - by Presence of Aneurysm')#年龄分布 - 按动脉瘤存在
#     plt.xlabel('age')
#     plt.ylabel('Frequency')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     age_groups = pd.cut(train_df['PatientAge'], bins=5)
#     pd.crosstab(age_groups, train_df['Aneurysm Present']).plot(kind='bar', figsize=(12, 6))
#     plt.title('The relationship between age groups and aneurysm presence')#年龄组与动脉瘤存在的关系
#     plt.xlabel('age group')#年龄组
#     plt.ylabel('Counting')
#     plt.legend(['Aneurysm-free', 'There is an aneurysm'])
#     plt.tight_layout()
#     plt.savefig('age_vs_aneurysm.png', dpi=300)
#     plt.show()

# # 成像模式与动脉瘤存在的关系
# if 'Modality' in train_df.columns and 'Aneurysm Present' in train_df.columns:
#     modality_aneurysm = pd.crosstab(train_df['Modality'], train_df['Aneurysm Present'])
#     print("\n成像模式与动脉瘤存在的关系:")
#     print(modality_aneurysm)
    
#     modality_aneurysm.plot(kind='bar', figsize=(10, 6))
#     plt.title('The relationship between imaging modes and aneurysm presence')#成像模式与动脉瘤存在的关系
#     plt.xlabel('Imaging mode')#成像模式
#     plt.ylabel('Counting')
#     plt.legend(['Aneurysm-free', 'There is an aneurysm'])
#     plt.savefig('modality_vs_aneurysm.png', dpi=300)
#     plt.show()

# # 各位置动脉瘤的相关性分析
# print("\n各位置动脉瘤的相关性矩阵:")
# location_corr = train_df[location_labels].corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(location_corr, annot=True, cmap='coolwarm', center=0)
# plt.title('Aneurysm-related heat map at various positions')#各位置动脉瘤相关性热图
# plt.tight_layout()
# plt.savefig('location_correlation.png', dpi=300)
# plt.show()

# # ===================== 数据集划分 =====================
# print("\n===== 数据集划分 =====")

# # 使用分层抽样确保各类别比例一致
# train_df, test_df = train_test_split(
#     train_df, 
#     test_size=0.2, 
#     random_state=42, 
#     stratify=train_df['Aneurysm Present']
# )
# train_df, val_df = train_test_split(
#     train_df, 
#     test_size=0.125,  # 0.8 * 0.125 = 0.1
#     random_state=42, 
#     stratify=train_df['Aneurysm Present']
# )

# print(f"训练集大小: {len(train_df)}")
# print(f"验证集大小: {len(val_df)}")
# print(f"测试集大小: {len(test_df)}")

# # 显示各数据集标签分布
# print("\n训练集动脉瘤存在分布:")
# print(train_df['Aneurysm Present'].value_counts(normalize=True))

# print("\n验证集动脉瘤存在分布:")
# print(val_df['Aneurysm Present'].value_counts(normalize=True))

# print("\n测试集动脉瘤存在分布:")
# print(test_df['Aneurysm Present'].value_counts(normalize=True))

# # ===================== 数据集类定义 =====================
# class AneurysmDataset(Dataset):
#     def __init__(self, df, root_dir='/kaggle/input/rsna-intracranial-aneurysm-detection/series', 
#                  transform=None, is_test=False, location_labels=None):
#         self.df = df.copy()
#         self.root_dir = root_dir
#         self.transform = transform
#         self.is_test = is_test
#         self.location_labels = location_labels
#         self.target_shape = (1, 256, 256)  # (通道数, 高度, 宽度)
#         self.label_shape = (14,)
        
#         self._validate_and_clean_labels()
        
#         if not os.path.exists(self.root_dir):
#             raise FileNotFoundError(f"根目录不存在: {self.root_dir}")
            
#         valid_indices = self._filter_valid_series()
#         self.df = self.df.iloc[valid_indices].reset_index(drop=True)
#         print(f"过滤后保留的有效样本数: {len(self.df)}")

#     def _validate_and_clean_labels(self):
#         if not self.location_labels:
#             raise ValueError("location_labels 不能为空")
            
#         all_label_cols = ['Aneurysm Present'] + self.location_labels
#         missing_cols = [col for col in all_label_cols if col not in self.df.columns]
#         if missing_cols:
#             raise KeyError(f"缺少标签列: {missing_cols}")
        
#         for col in all_label_cols:
#             self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
#             self.df[col].fillna(0.0, inplace=True)
#             self.df[col] = self.df[col].astype(np.float32)

#     def _filter_valid_series(self):
#         valid_indices = []
#         for idx in tqdm(range(len(self.df)), desc="过滤有效样本"):
#             series_id = self.df.iloc[idx]['SeriesInstanceUID']
#             series_path = os.path.join(self.root_dir, series_id)
#             if os.path.exists(series_path) and len(os.listdir(series_path)) > 0:
#                 valid_indices.append(idx)
#             else:
#                 warn_msg = f"路径不存在或为空 - {series_path}"
#                 logging.warning(warn_msg)
#         return valid_indices

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         series_id = self.df.iloc[idx]['SeriesInstanceUID']
#         series_path = os.path.join(self.root_dir, series_id)
        
#         # 初始化返回值
#         image = torch.zeros(self.target_shape, dtype=torch.float32)
#         label = torch.zeros(self.label_shape, dtype=torch.float32)

#         try:
#             # 加载DICOM文件列表
#             dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
#             if not dicom_files:
#                 warn_msg = f"未找到DICOM文件 - {series_path}"
#                 logging.warning(warn_msg)
#                 return image, label
            
#             # 按数字排序并选择中间切片
#             dicom_files.sort(key=lambda x: int(x.split('.')[0]))
#             mid_idx = len(dicom_files) // 2
#             dicom_path = os.path.join(series_path, dicom_files[mid_idx])
            
#             # 读取DICOM文件
#             dicom = pydicom.dcmread(dicom_path)
            
#             # 应用VOI LUT增强对比度
#             pixel_array = apply_voi_lut(dicom.pixel_array, dicom)
            
#             # 处理3D数据
#             if len(pixel_array.shape) == 3:
#                 slice_idx = pixel_array.shape[0] // 2
#                 pixel_array = pixel_array[slice_idx, :, :]
#                 logging.warning(f"{series_id} 是3D数据，已取中间切片")
#             elif len(pixel_array.shape) != 2:
#                 error_msg = f"DICOM图像维度异常: {pixel_array.shape}，series_id: {series_id}"
#                 logging.error(error_msg)
#                 return image, label
            
#             # 转换为float32并应用 rescale 校正
#             pixel_array = pixel_array.astype(np.float32)
#             if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
#                 pixel_array = pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept
            
#             # 标准化到0-1范围
#             img_min, img_max = pixel_array.min(), pixel_array.max()
#             if img_max != img_min:
#                 pixel_array = (pixel_array - img_min) / (img_max - img_min)
            
#             # 转换为张量并添加通道维度
#             img_tensor = torch.from_numpy(pixel_array).unsqueeze(0)
            
#             # 调整尺寸
#             resize_transform = transforms.Resize((self.target_shape[1], self.target_shape[2]))
#             img_tensor = resize_transform(img_tensor)
            
#             if img_tensor.shape != self.target_shape:
#                 raise RuntimeError(f"图像形状错误: {img_tensor.shape} 预期 {self.target_shape}")
            
#             image = img_tensor
            
#         except Exception as e:
#             error_msg = f"处理DICOM时出错 {series_path}: {str(e)}"
#             logging.error(error_msg)
        
#         # 处理标签
#         try:
#             if not self.is_test:
#                 present = float(self.df.iloc[idx]['Aneurysm Present'])
#                 locations = self.df.iloc[idx][self.location_labels].values.astype(np.float32)
#                 label_np = np.concatenate([[present], locations])
#                 label = torch.FloatTensor(label_np)
#         except Exception as e:
#             error_msg = f"处理标签时出错 {series_id}: {str(e)}"
#             logging.error(error_msg)
        
#         # 应用变换
#         if self.transform:
#             try:
#                 image = self.transform(image)
#             except Exception as e:
#                 logging.error(f"应用变换时出错: {str(e)}")
        
#         return image, label

# # ===================== 数据加载器准备 =====================
# print("\n===== 准备数据加载器 =====")

# # 定义图像变换 - 增加数据增强提高模型泛化能力
# train_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
#     transforms.RandomHorizontalFlip(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# # 创建数据集实例
# train_dataset = AneurysmDataset(
#     train_df, 
#     transform=train_transform, 
#     location_labels=location_labels
# )
# val_dataset = AneurysmDataset(
#     val_df, 
#     transform=val_transform, 
#     location_labels=location_labels
# )
# test_dataset = AneurysmDataset(
#     test_df, 
#     transform=val_transform, 
#     is_test=False,  # 保持为False以便评估
#     location_labels=location_labels
# )

# # 创建数据加载器
# batch_size = 16
# num_workers = 0 if device.type == 'cpu' else 2
# train_loader = DataLoader(
#     train_dataset, 
#     batch_size=batch_size, 
#     shuffle=True, 
#     num_workers=num_workers,
#     pin_memory=True if device.type == 'cuda' else False
# )
# val_loader = DataLoader(
#     val_dataset, 
#     batch_size=batch_size, 
#     shuffle=False, 
#     num_workers=num_workers,
#     pin_memory=True if device.type == 'cuda' else False
# )
# test_loader = DataLoader(
#     test_dataset, 
#     batch_size=batch_size, 
#     shuffle=False, 
#     num_workers=num_workers,
#     pin_memory=True if device.type == 'cuda' else False
# )

# # ===================== 模型定义 =====================
# class AneurysmViT(nn.Module):
#     def __init__(self, num_labels=14):
#         super(AneurysmViT, self).__init__()
        
#         self.vit_config = ViTConfig(
#             image_size=256,
#             patch_size=32,
#             num_channels=1,
#             num_labels=num_labels,
#             hidden_size=768,
#             num_hidden_layers=12,
#             num_attention_heads=12,
#             intermediate_size=3072,
#             dropout=0.3,  # 增加dropout防止过拟合
#             attention_probs_dropout_prob=0.3
#         )
        
#         self.vit = ViTModel(self.vit_config)
        
#         # 改进分类头
#         self.classifier = nn.Sequential(
#             nn.Linear(768, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, num_labels)
#         )
        
#     def forward(self, x):
#         outputs = self.vit(pixel_values=x)
#         cls_token = outputs.last_hidden_state[:, 0, :]
#         logits = self.classifier(cls_token)
#         return torch.sigmoid(logits)

# # 初始化模型
# model = AneurysmViT().to(device)
# print("\n模型结构:")
# print(model)

# # ===================== 训练配置 =====================
# # 计算类别权重 - 更精确地处理类别不平衡
# present_count = train_df['Aneurysm Present'].sum()
# total_count = len(train_df)
# weight_positive = total_count / (2 * present_count) if present_count > 0 else 1.0
# weight_negative = total_count / (2 * (total_count - present_count)) if (total_count - present_count) > 0 else 1.0

# print(f"\n类别权重 - 阳性: {weight_positive:.2f}, 阴性: {weight_negative:.2f}")

# # 加权损失函数
# class WeightedBCELoss(nn.Module):
#     def __init__(self, pos_weight=1.0):
#         super().__init__()
#         self.pos_weight = pos_weight
        
#     def forward(self, output, target):
#         # 对第一个标签（是否存在动脉瘤）应用特殊权重
#         weights = torch.ones_like(target, device=device)
#         weights[:, 0] = self.pos_weight
        
#         bce_loss = nn.BCELoss(reduction='none')(output, target)
#         return (bce_loss * weights).mean()

# # 优化器和调度器
# criterion = WeightedBCELoss(pos_weight=weight_positive)
# optimizer = torch.optim.AdamW(
#     model.parameters(), 
#     lr=1e-4, 
#     weight_decay=1e-5
# )
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, 
#     T_0=5, 
#     T_mult=2, 
#     eta_min=1e-6
# )

# # 训练参数
# num_epochs = 15
# best_val_score = 0.0
# early_stop_patience = 5
# early_stop_counter = 0

# # 记录训练过程指标
# train_losses = []
# val_losses = []
# train_aucs = []
# val_aucs = []

# # ===================== 训练循环 =====================
# print("\n===== 开始训练 =====")
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0.0
#     train_auc = 0.0
#     train_auc_count = 0
    
#     # 训练循环带进度条
#     train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 训练")
#     for images, labels in train_pbar:
#         images = images.to(device)
#         labels = labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
        
#         # 梯度裁剪防止梯度爆炸
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         optimizer.step()
        
#         train_loss += loss.item() * images.size(0)
        
#         # 计算AUC
#         labels_np = labels.cpu().detach().numpy()
#         outputs_np = outputs.cpu().detach().numpy()
        
#         try:
#             if len(np.unique(labels_np[:, 0])) >= 2:
#                 auc = roc_auc_score(labels_np[:, 0], outputs_np[:, 0])
#                 train_auc += auc * images.size(0)
#                 train_auc_count += images.size(0)
#         except ValueError as e:
#             logging.error(f"训练AUC计算错误: {str(e)}")
        
#         # 更新进度条
#         train_pbar.set_postfix({"batch_loss": loss.item()})
    
#     # 计算平均训练损失和AUC
#     train_loss /= len(train_loader.dataset)
#     train_auc = train_auc / train_auc_count if train_auc_count > 0 else 0.5
    
#     # 验证阶段
#     model.eval()
#     val_loss = 0.0
#     val_scores = []
#     val_main_aucs = []
    
#     with torch.no_grad():
#         val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 验证")
#         for images, labels in val_pbar:
#             images = images.to(device)
#             labels = labels.to(device)
            
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item() * images.size(0)
            
#             # 计算AUC
#             labels_np = labels.cpu().numpy()
#             outputs_np = outputs.cpu().numpy()
#             auc_scores = []
            
#             for i in range(labels_np.shape[1]):
#                 try:
#                     if len(np.unique(labels_np[:, i])) >= 2:
#                         auc_scores.append(roc_auc_score(labels_np[:, i], outputs_np[:, i]))
#                     else:
#                         auc_scores.append(0.5)
#                 except:
#                     auc_scores.append(0.5)
            
#             main_auc = auc_scores[0]
#             other_auc = np.mean(auc_scores[1:])
#             val_scores.append(0.5 * (main_auc + other_auc))
#             val_main_aucs.append(main_auc)
            
#             # 更新进度条
#             val_pbar.set_postfix({"batch_loss": loss.item()})
    
#     # 计算平均验证损失和分数
#     val_loss /= len(val_loader.dataset)
#     val_score = np.mean(val_scores)
#     val_main_auc = np.mean(val_main_aucs)
    
#     # 更新学习率调度器
#     scheduler.step()
    
#     # 记录指标
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)
#     train_aucs.append(train_auc)
#     val_aucs.append(val_main_auc)
    
#     # 保存最佳模型
#     if val_score > best_val_score:
#         best_val_score = val_score
#         torch.save(model.state_dict(), 'best_model.pth')
#         early_stop_counter = 0
#         print("保存最佳模型!")
#     else:
#         early_stop_counter += 1
#         if early_stop_counter >= early_stop_patience:
#             print("早停: 验证分数连续多轮无提升")
#             break
    
#     # 打印 epoch 结果
#     print(f"\nEpoch {epoch+1}/{num_epochs}")
#     print(f"训练损失: {train_loss:.4f} | 训练主标签AUC: {train_auc:.4f}")
#     print(f"验证损失: {val_loss:.4f} | 验证主标签AUC: {val_main_auc:.4f} | 验证综合分数: {val_score:.4f}")
#     print("-" * 80)

# print("训练完成!")

# # ===================== 绘制训练历史 =====================
# print("\n绘制训练历史...")
# plt.figure(figsize=(14, 6))

# # 损失曲线
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='训练损失')
# plt.plot(val_losses, label='验证损失')
# plt.title('Training and Validation Loss Curve')#训练和验证损失曲线
# plt.xlabel('Epoch')
# plt.ylabel('Loss Value')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)

# # AUC曲线
# plt.subplot(1, 2, 2)
# plt.plot(train_aucs, label='训练AUC')
# plt.plot(val_aucs, label='验证AUC')
# plt.title('Training and Validation AUC Curve')#训练和验证AUC曲线
# plt.xlabel('Epoch')
# plt.ylabel('AUC value')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.savefig('training_history.png', dpi=300)
# plt.show()

# # ===================== 测试集评估 =====================
# print("\n===== 测试集评估 =====")
# # 加载最佳模型
# model.load_state_dict(torch.load('best_model.pth'))
# model.eval()

# test_scores = []
# test_aucs = []
# all_auc_scores = []

# with torch.no_grad():
#     test_pbar = tqdm(test_loader, desc="测试集评估")
#     for images, labels in test_pbar:
#         images = images.to(device)
#         labels = labels.to(device)
        
#         outputs = model(images)
        
#         # 计算AUC
#         labels_np = labels.cpu().numpy()
#         outputs_np = outputs.cpu().numpy()
        
#         # 计算每个标签的AUC
#         auc_scores = []
#         for i in range(labels_np.shape[1]):
#             try:
#                 auc = roc_auc_score(labels_np[:, i], outputs_np[:, i])
#                 auc_scores.append(auc)
#             except ValueError:
#                 auc_scores.append(0.5)
        
#         all_auc_scores.append(auc_scores)
        
#         # 计算加权分数
#         main_auc = auc_scores[0]
#         other_auc = np.mean(auc_scores[1:])
#         weighted_score = 0.5 * (main_auc + other_auc)
        
#         test_scores.append(weighted_score)
#         test_aucs.append(main_auc)

# # 计算平均分数
# test_score = np.mean(test_scores)
# test_auc = np.mean(test_aucs)

# print(f"\n测试集主标签AUC: {test_auc:.4f}")
# print(f"测试集综合分数: {test_score:.4f}")

# # 计算每个位置的平均AUC
# mean_auc_per_label = np.mean(all_auc_scores, axis=0)
# print("\n每个标签的平均AUC:")
# for i, label in enumerate(['Aneurysm Present'] + location_labels):
#     print(f"{label}: {mean_auc_per_label[i]:.4f}")

# # ===================== 可视化预测结果 =====================
# print("\n可视化预测结果...")
# def visualize_predictions(model, dataset, num_samples=5):
#     model.eval()
#     indices = np.random.choice(len(dataset), num_samples, replace=False)
    
#     plt.figure(figsize=(15, 4*num_samples))
#     for i, idx in enumerate(indices):
#         image, label = dataset[idx]
#         image_tensor = image.unsqueeze(0).to(device)
        
#         with torch.no_grad():
#             pred = model(image_tensor).cpu().numpy()[0]
        
#         # 显示图像
#         plt.subplot(num_samples, 2, 2*i+1)
#         plt.imshow(image[0], cmap='gray')
#         plt.title(f"真实值: {label[0].item():.0f}, 预测值: {pred[0]:.2f}")
#         plt.axis('off')
        
#         # 显示位置预测
#         plt.subplot(num_samples, 2, 2*i+2)
#         plt.barh(location_labels, pred[1:])
#         plt.xlim(0, 1)
#         plt.title("Position Prediction Probability")#位置预测概率
    
#     plt.tight_layout()
#     plt.savefig('predictions_visualization.png', dpi=300)
#     plt.show()

# # 可视化测试集预测结果
# visualize_predictions(model, test_dataset, num_samples=5)

# # ===================== 生成提交文件 =====================
# # print("\n生成提交文件...")
# # def generate_submission(model, test_loader, dataset, device):
# #     model.eval()
# #     all_preds = []
# #     all_ids = []
    
# #     with torch.no_grad():
# #         pbar = tqdm(test_loader, desc="生成预测结果")
# #         for images, _ in pbar:
# #             images = images.to(device)
# #             outputs = model(images)
# #             preds = outputs.cpu().numpy()
# #             all_preds.append(preds)
    
# #     # 获取真实的SeriesInstanceUID
# #     all_ids = dataset.df['SeriesInstanceUID'].values
    
# #     all_preds = np.concatenate(all_preds)
    
# #     # 创建提交DataFrame
# #     submission_df = pd.DataFrame(all_preds, columns=['Aneurysm Present'] + location_labels)
# #     submission_df.insert(0, 'SeriesInstanceUID', all_ids)
    
# #     return submission_df

# # # 生成提交文件
# # submission_df = generate_submission(model, test_loader, test_dataset, device)
# # submission_df.to_csv('submission.csv', index=False)
# # print("提交文件已保存为 submission.csv")

# # print("\n===== 代码执行完成 =====")
# print("\n生成提交文件...")
# def generate_submission(model, test_loader, dataset, device):
#     model.eval()
#     all_preds = []
#     all_ids = []
    
#     with torch.no_grad():
#         pbar = tqdm(test_loader, desc="生成预测结果")
#         for images, _ in pbar:
#             images = images.to(device)
#             outputs = model(images)
#             preds = outputs.cpu().numpy()
#             all_preds.append(preds)
    
#     # 获取真实的SeriesInstanceUID
#     all_ids = dataset.df['SeriesInstanceUID'].values
    
#     all_preds = np.concatenate(all_preds)
    
#     # 创建提交DataFrame
#     submission_df = pd.DataFrame(all_preds, columns=['Aneurysm Present'] + location_labels)
#     submission_df.insert(0, 'SeriesInstanceUID', all_ids)
    
#     return submission_df

# # 生成提交文件（保存为parquet格式，替换原csv逻辑）
# submission_df = generate_submission(model, test_loader, test_dataset, device)
# # 使用snappy压缩减少文件体积，兼容大多数比赛平台
# submission_df.to_parquet('submission.parquet', index=False, compression='snappy')

# # 验证文件是否生成成功
# if os.path.exists('submission.parquet'):
#     file_size = os.path.getsize('submission.parquet') / 1024 / 1024  # 转换为MB
#     print(f"提交文件已成功保存为 submission.parquet，文件大小: {file_size:.2f} MB")
# else:
#     print("错误：提交文件未生成")

# print("\n===== 代码执行完成 =====")


# import os
# import numpy as np
# import pandas as pd
# import pydicom
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from transformers import ViTModel, ViTConfig
# from safetensors.torch import load_file
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import warnings
# import logging
# from pydicom.pixel_data_handlers.util import apply_voi_lut

# # 配置日志记录
# logging.basicConfig(
#     filename='dicom_processing.log',
#     level=logging.WARNING,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# warnings.filterwarnings('ignore')

# # 设置随机种子保证可重复性
# torch.manual_seed(42)
# np.random.seed(42)

# # 检查GPU可用性
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"使用设备: {device}")

# # 定义动脉瘤位置标签
# location_labels = [
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
#     'Other Posterior Circulation'
# ]
# all_label_cols = ['Aneurysm Present'] + location_labels

# # ===================== 数据集类定义 =====================
# class AneurysmDataset(Dataset):
#     def __init__(self, df, root_dir='/kaggle/input/rsna-intracranial-aneurysm-detection/series', 
#                  transform=None, is_test=False, location_labels=None):
#         self.df = df.copy()
#         self.root_dir = root_dir
#         self.transform = transform
#         self.is_test = is_test
#         self.location_labels = location_labels
#         self.target_shape = (1, 224, 224)  # 匹配模型输入尺寸224x224
#         self.label_shape = (14,)
        
#         self._validate_and_clean_labels()
        
#         if not os.path.exists(self.root_dir):
#             raise FileNotFoundError(f"根目录不存在: {self.root_dir}")
            
#         valid_indices = self._filter_valid_series()
#         self.df = self.df.iloc[valid_indices].reset_index(drop=True)
#         print(f"过滤后保留的有效样本数: {len(self.df)}")

#     def _validate_and_clean_labels(self):
#         if not self.location_labels:
#             raise ValueError("location_labels 不能为空")
            
#         all_label_cols = ['Aneurysm Present'] + self.location_labels
#         missing_cols = [col for col in all_label_cols if col not in self.df.columns]
#         if missing_cols:
#             raise KeyError(f"缺少标签列: {missing_cols}")
        
#         for col in all_label_cols:
#             self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
#             self.df[col].fillna(0.0, inplace=True)
#             self.df[col] = self.df[col].astype(np.float32)

#     def _filter_valid_series(self):
#         valid_indices = []
#         for idx in tqdm(range(len(self.df)), desc="过滤有效样本"):
#             series_id = self.df.iloc[idx]['SeriesInstanceUID']
#             series_path = os.path.join(self.root_dir, series_id)
#             if os.path.exists(series_path) and len(os.listdir(series_path)) > 0:
#                 valid_indices.append(idx)
#             else:
#                 warn_msg = f"路径不存在或为空 - {series_path}"
#                 logging.warning(warn_msg)
#         return valid_indices

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         series_id = self.df.iloc[idx]['SeriesInstanceUID']
#         series_path = os.path.join(self.root_dir, series_id)
        
#         # 初始化返回值
#         image = torch.zeros(self.target_shape, dtype=torch.float32)
#         label = torch.zeros(self.label_shape, dtype=torch.float32)

#         try:
#             # 加载DICOM文件列表
#             dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
#             if not dicom_files:
#                 warn_msg = f"未找到DICOM文件 - {series_path}"
#                 logging.warning(warn_msg)
#                 return image, label
            
#             # 按数字排序并选择中间切片
#             dicom_files.sort(key=lambda x: int(x.split('.')[0]))
#             mid_idx = len(dicom_files) // 2
#             dicom_path = os.path.join(series_path, dicom_files[mid_idx])
            
#             # 读取DICOM文件
#             dicom = pydicom.dcmread(dicom_path)
            
#             # 应用VOI LUT增强对比度
#             pixel_array = apply_voi_lut(dicom.pixel_array, dicom)
            
#             # 处理3D数据
#             if len(pixel_array.shape) == 3:
#                 slice_idx = pixel_array.shape[0] // 2
#                 pixel_array = pixel_array[slice_idx, :, :]
#                 logging.warning(f"{series_id} 是3D数据，已取中间切片")
#             elif len(pixel_array.shape) != 2:
#                 error_msg = f"DICOM图像维度异常: {pixel_array.shape}，series_id: {series_id}"
#                 logging.error(error_msg)
#                 return image, label
            
#             # 转换为float32并应用 rescale 校正
#             pixel_array = pixel_array.astype(np.float32)
#             if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
#                 pixel_array = pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept
            
#             # 标准化到0-1范围
#             img_min, img_max = pixel_array.min(), pixel_array.max()
#             if img_max != img_min:
#                 pixel_array = (pixel_array - img_min) / (img_max - img_min)
            
#             # 转换为张量并添加通道维度
#             img_tensor = torch.from_numpy(pixel_array).unsqueeze(0)
            
#             # 调整尺寸到224x224
#             resize_transform = transforms.Resize((self.target_shape[1], self.target_shape[2]))
#             img_tensor = resize_transform(img_tensor)
            
#             if img_tensor.shape != self.target_shape:
#                 raise RuntimeError(f"图像形状错误: {img_tensor.shape} 预期 {self.target_shape}")
            
#             image = img_tensor
            
#         except Exception as e:
#             error_msg = f"处理DICOM时出错 {series_path}: {str(e)}"
#             logging.error(error_msg)
        
#         # 处理标签
#         try:
#             if not self.is_test:
#                 present = float(self.df.iloc[idx]['Aneurysm Present'])
#                 locations = self.df.iloc[idx][self.location_labels].values.astype(np.float32)
#                 label_np = np.concatenate([[present], locations])
#                 label = torch.FloatTensor(label_np)
#         except Exception as e:
#             error_msg = f"处理标签时出错 {series_id}: {str(e)}"
#             logging.error(error_msg)
        
#         # 应用变换
#         if self.transform:
#             try:
#                 image = self.transform(image)
#             except Exception as e:
#                 logging.error(f"应用变换时出错: {str(e)}")
        
#         return image, label

# # ===================== 模型定义 =====================
# class AneurysmViT(nn.Module):
#     def __init__(self, num_labels=14):
#         super(AneurysmViT, self).__init__()
        
#         # 配置与预训练权重完全匹配（224x224图像、16x16 patch、3通道）
#         self.vit_config = ViTConfig(
#             image_size=224,
#             patch_size=16,
#             num_channels=3,
#             hidden_size=768,
#             num_hidden_layers=12,
#             num_attention_heads=12,
#             intermediate_size=3072,
#             dropout=0.3,
#             attention_probs_dropout_prob=0.3
#         )
        
#         # 直接用配置初始化ViT模型
#         self.vit = ViTModel(self.vit_config)
        
#         # 加载外部预训练权重
#         model_path = '/kaggle/input/vit/transformers/default/1/vit_model/model.safetensors'
#         if os.path.exists(model_path):
#             print(f"正在加载预训练模型: {model_path}")
#             state_dict = load_file(model_path)
            
#             # 清理权重键名（移除可能的"vit."前缀）
#             cleaned_state_dict = {}
#             for param_name, param_value in state_dict.items():
#                 if param_name.startswith('vit.'):
#                     cleaned_name = param_name[4:]
#                 else:
#                     cleaned_name = param_name
#                 cleaned_state_dict[cleaned_name] = param_value
            
#             # 加载权重（strict=True，确保完全匹配）
#             self.vit.load_state_dict(cleaned_state_dict, strict=True)
#             print("预训练权重加载成功（结构完全匹配）")
#         else:
#             raise FileNotFoundError(f"预训练模型文件不存在: {model_path}")
        
#         # 冻结部分预训练层
#         freeze_layers = 8  # 冻结前8层
#         for i, (name, param) in enumerate(self.vit.named_parameters()):
#             if i < freeze_layers * 12:  # 每层约12个参数组
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True
        
#         # 分类头
#         self.classifier = nn.Sequential(
#             nn.Linear(768, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, num_labels)
#         )
        
#     def forward(self, x):
#         # 1通道灰度图 → 3通道（复制3次匹配预训练输入）
#         x = x.repeat(1, 3, 1, 1)  # 形状: [batch, 1, 224, 224] → [batch, 3, 224, 224]
        
#         # ViT前向传播
#         vit_outputs = self.vit(pixel_values=x)
#         cls_token = vit_outputs.last_hidden_state[:, 0, :]
        
#         # 分类头预测
#         logits = self.classifier(cls_token)
#         return torch.sigmoid(logits)

# # ===================== 主函数 =====================
# def main():
#     # 1. 数据加载与清洗
#     print("===== 1. 数据加载与清洗 =====")
#     train_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
#     localizers_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')
    
#     print(f"原始训练数据形状: {train_df.shape}")
#     print(f"定位器数据形状: {localizers_df.shape}")
    
#     # 处理缺失值
#     print(f"\n预处理前缺失值总数: {train_df.isnull().sum().sum()}")
#     train_df.fillna(0, inplace=True)
#     localizers_df.dropna(inplace=True)
#     print(f"预处理后缺失值总数: {train_df.isnull().sum().sum()}")
    
#     # 统一标签类型
#     for col in all_label_cols:
#         if col in train_df.columns:
#             if train_df[col].dtype == 'object':
#                 train_df[col] = train_df[col].apply(lambda x: 
#                     1.0 if str(x).strip().lower() in ['true', 'yes', 'present', '1'] else
#                     0.0 if str(x).strip().lower() in ['false', 'no', 'absent', '0'] else
#                     float(x) if pd.notna(x) and str(x).strip() else 0.0
#                 )
#             train_df[col] = train_df[col].astype(np.float32)

#     # 2. 数据集划分
#     print("\n===== 2. 数据集划分 =====")
#     train_df, test_df = train_test_split(
#         train_df, 
#         test_size=0.2, 
#         random_state=42, 
#         stratify=train_df['Aneurysm Present']
#     )
#     train_df, val_df = train_test_split(
#         train_df, 
#         test_size=0.125,  # 0.8×0.125=0.1
#         random_state=42, 
#         stratify=train_df['Aneurysm Present']
#     )
    
#     print(f"训练集大小: {len(train_df)}")
#     print(f"验证集大小: {len(val_df)}")
#     print(f"测试集大小: {len(test_df)}")
    
#     # 验证分层效果
#     print("\n各数据集动脉瘤存在比例:")
#     print(f"训练集: {train_df['Aneurysm Present'].mean():.4f}")
#     print(f"验证集: {val_df['Aneurysm Present'].mean():.4f}")
#     print(f"测试集: {test_df['Aneurysm Present'].mean():.4f}")

#     # 3. 数据加载器准备
#     print("\n===== 3. 数据加载器准备 =====")
#     # 训练集数据增强
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])
    
#     # 验证集/测试集无增强
#     val_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])
    
#     # 创建数据集实例
#     train_dataset = AneurysmDataset(
#         train_df, 
#         transform=train_transform, 
#         location_labels=location_labels
#     )
#     val_dataset = AneurysmDataset(
#         val_df, 
#         transform=val_transform, 
#         location_labels=location_labels
#     )
#     test_dataset = AneurysmDataset(
#         test_df, 
#         transform=val_transform, 
#         is_test=False,
#         location_labels=location_labels
#     )
    
#     # 创建数据加载器
#     batch_size = 16
#     num_workers = 2 if device.type == 'cuda' else 0
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True if device.type == 'cuda' else False
#     )
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=batch_size, 
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True if device.type == 'cuda' else False
#     )
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=batch_size, 
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True if device.type == 'cuda' else False
#     )

#     # 4. 模型初始化与配置
#     print("\n===== 4. 模型初始化 =====")
#     model = AneurysmViT().to(device)
#     print("模型结构概览:")
#     print(model)
    
#     # 计算模型参数
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"总参数: {total_params:,} | 可训练参数: {trainable_params:,}")

#     # 5. 训练配置
#     print("\n===== 5. 训练配置 =====")
#     # 类别权重计算
#     present_count = train_df['Aneurysm Present'].sum()
#     total_count = len(train_df)
#     weight_positive = total_count / (2 * present_count) if present_count > 0 else 1.0
#     print(f"动脉瘤存在样本占比: {present_count/total_count:.4f}")
#     print(f"类别权重（阳性样本）: {weight_positive:.2f}")
    
#     # 自定义加权BCE损失
#     class WeightedBCELoss(nn.Module):
#         def __init__(self, pos_weight=1.0):
#             super().__init__()
#             self.pos_weight = pos_weight
            
#         def forward(self, output, target):
#             weights = torch.ones_like(target, device=device)
#             weights[:, 0] = self.pos_weight  # 对主标签加权
#             bce_loss = nn.BCELoss(reduction='none')(output, target)
#             return (bce_loss * weights).mean()
    
#     # 优化器和调度器
#     optimizer = torch.optim.AdamW(
#         model.parameters(), 
#         lr=1e-4, 
#         weight_decay=1e-5
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#         optimizer, 
#         T_0=5, 
#         T_mult=2, 
#         eta_min=1e-6
#     )

#     # 6. 模型训练与验证
#     print("\n===== 6. 开始训练 =====")
#     num_epochs = 15
#     best_val_score = 0.0
#     early_stop_patience = 5
#     early_stop_counter = 0
    
#     # 记录训练指标
#     train_losses = []
#     val_losses = []
#     train_aucs = []
#     val_aucs = []
    
#     for epoch in range(num_epochs):
#         # 训练阶段
#         model.train()
#         train_loss = 0.0
#         train_auc = 0.0
#         train_auc_count = 0
        
#         train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 训练")
#         for images, labels in train_pbar:
#             images = images.to(device)
#             labels = labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = WeightedBCELoss(pos_weight=weight_positive)(outputs, labels)
#             loss.backward()
            
#             # 梯度裁剪
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
            
#             train_loss += loss.item() * images.size(0)
            
#             # 计算AUC
#             labels_np = labels.cpu().detach().numpy()
#             outputs_np = outputs.cpu().detach().numpy()
            
#             try:
#                 if len(np.unique(labels_np[:, 0])) >= 2:
#                     auc = roc_auc_score(labels_np[:, 0], outputs_np[:, 0])
#                     train_auc += auc * images.size(0)
#                     train_auc_count += images.size(0)
#             except ValueError as e:
#                 logging.error(f"训练AUC计算错误: {str(e)}")
            
#             train_pbar.set_postfix({"batch_loss": loss.item()})
        
#         # 计算训练集平均指标
#         train_loss /= len(train_loader.dataset)
#         train_auc = train_auc / train_auc_count if train_auc_count > 0 else 0.5
        
#         # 验证阶段
#         model.eval()
#         val_loss = 0.0
#         val_scores = []
#         val_main_aucs = []
        
#         with torch.no_grad():
#             val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 验证")
#             for images, labels in val_pbar:
#                 images = images.to(device)
#                 labels = labels.to(device)
                
#                 outputs = model(images)
#                 loss = WeightedBCELoss(pos_weight=weight_positive)(outputs, labels)
#                 val_loss += loss.item() * images.size(0)
                
#                 # 计算AUC
#                 labels_np = labels.cpu().numpy()
#                 outputs_np = outputs.cpu().numpy()
#                 auc_scores = []
                
#                 for i in range(labels_np.shape[1]):
#                     try:
#                         if len(np.unique(labels_np[:, i])) >= 2:
#                             auc_scores.append(roc_auc_score(labels_np[:, i], outputs_np[:, i]))
#                         else:
#                             auc_scores.append(0.5)
#                     except:
#                         auc_scores.append(0.5)
                
#                 main_auc = auc_scores[0]
#                 other_auc = np.mean(auc_scores[1:])
#                 val_scores.append(0.5 * (main_auc + other_auc))
#                 val_main_aucs.append(main_auc)
                
#                 val_pbar.set_postfix({"batch_loss": loss.item()})
        
#         # 计算验证集平均指标
#         val_loss /= len(val_loader.dataset)
#         val_score = np.mean(val_scores)
#         val_main_auc = np.mean(val_main_aucs)
        
#         # 更新调度器
#         scheduler.step()
        
#         # 记录指标
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         train_aucs.append(train_auc)
#         val_aucs.append(val_main_auc)
        
#         # 保存最佳模型
#         if val_score > best_val_score:
#             best_val_score = val_score
#             torch.save(model.state_dict(), 'best_model.pth')
#             early_stop_counter = 0
#             print(f"保存最佳模型（验证综合评分: {val_score:.4f}）")
#         else:
#             early_stop_counter += 1
#             if early_stop_counter >= early_stop_patience:
#                 print(f"早停触发：连续{early_stop_patience}轮验证分数无提升")
#                 break
        
#         # 打印本轮结果
#         print(f"\nEpoch {epoch+1}/{num_epochs} 结果:")
#         print(f"训练损失: {train_loss:.4f} | 训练主标签AUC: {train_auc:.4f}")
#         print(f"验证损失: {val_loss:.4f} | 验证主标签AUC: {val_main_auc:.4f} | 验证综合评分: {val_score:.4f}")
#         print("-" * 80)
    
#     print("训练完成!")

#     # 7. 训练历史可视化
#     print("\n===== 7. 训练历史可视化 =====")
#     plt.figure(figsize=(14, 6))
    
#     # 损失曲线
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses, label='训练损失')
#     plt.plot(val_losses, label='验证损失')
#     plt.title('训练和验证损失曲线')
#     plt.xlabel('Epoch')
#     plt.ylabel('损失值')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # AUC曲线
#     plt.subplot(1, 2, 2)
#     plt.plot(train_aucs, label='训练AUC')
#     plt.plot(val_aucs, label='验证AUC')
#     plt.title('训练和验证AUC曲线')
#     plt.xlabel('Epoch')
#     plt.ylabel('AUC值')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     plt.tight_layout()
#     plt.savefig('training_history.png', dpi=300)
#     plt.show()

#     # 8. 测试集评估
#     print("\n===== 8. 测试集评估 =====")
#     # 加载最佳模型
#     model.load_state_dict(torch.load('best_model.pth'))
#     model.eval()
    
#     test_scores = []
#     test_aucs = []
#     all_auc_scores = []
    
#     with torch.no_grad():
#         test_pbar = tqdm(test_loader, desc="测试集评估")
#         for images, labels in test_pbar:
#             images = images.to(device)
#             labels = labels.to(device)
            
#             outputs = model(images)
            
#             # 计算AUC
#             labels_np = labels.cpu().numpy()
#             outputs_np = outputs.cpu().numpy()
#             auc_scores = []
            
#             for i in range(labels_np.shape[1]):
#                 try:
#                     auc = roc_auc_score(labels_np[:, i], outputs_np[:, i])
#                     auc_scores.append(auc)
#                 except ValueError:
#                     auc_scores.append(0.5)
            
#             all_auc_scores.append(auc_scores)
            
#             # 计算综合评分
#             main_auc = auc_scores[0]
#             other_auc = np.mean(auc_scores[1:])
#             weighted_score = 0.5 * (main_auc + other_auc)
            
#             test_scores.append(weighted_score)
#             test_aucs.append(main_auc)
    
#     # 计算测试集最终指标
#     test_score = np.mean(test_scores)
#     test_auc = np.mean(test_aucs)
    
#     print(f"\n测试集结果:")
#     print(f"主标签（是否存在动脉瘤）AUC: {test_auc:.4f}")
#     print(f"综合评分（主标签+位置标签）: {test_score:.4f}")
    
#     # 打印各位置标签的AUC
#     print("\n各标签AUC值:")
#     mean_auc_per_label = np.mean(all_auc_scores, axis=0)
#     for i, label in enumerate(['Aneurysm Present'] + location_labels):
#         print(f"{label}: {mean_auc_per_label[i]:.4f}")

#     # 9. 预测结果可视化
#     print("\n===== 9. 预测结果可视化 =====")
#     def visualize_predictions(model, dataset, num_samples=5):
#         model.eval()
#         indices = np.random.choice(len(dataset), num_samples, replace=False)
        
#         plt.figure(figsize=(15, 4*num_samples))
#         for i, idx in enumerate(indices):
#             image, label = dataset[idx]
#             image_tensor = image.unsqueeze(0).to(device)
            
#             with torch.no_grad():
#                 pred = model(image_tensor).cpu().numpy()[0]
            
#             # 显示医学图像
#             plt.subplot(num_samples, 2, 2*i+1)
#             plt.imshow(image[0], cmap='gray')
#             plt.title(f"真实值: {label[0].item():.0f}, 预测值: {pred[0]:.2f}")
#             plt.axis('off')
            
#             # 显示位置预测概率
#             plt.subplot(num_samples, 2, 2*i+2)
#             plt.barh(location_labels, pred[1:])
#             plt.xlim(0, 1)
#             plt.title("位置预测概率")
        
#         plt.tight_layout()
#         plt.savefig('predictions_visualization.png', dpi=300)
#         plt.show()
    
#     # 可视化测试集预测结果
#     visualize_predictions(model, test_dataset, num_samples=5)

#     # 10. 生成提交文件
#     print("\n===== 10. 生成提交文件 =====")
#     def generate_submission(model, test_loader, dataset, device):
#         model.eval()
#         all_preds = []
#         all_ids = dataset.df['SeriesInstanceUID'].values
        
#         with torch.no_grad():
#             pbar = tqdm(test_loader, desc="生成提交结果")
#             for images, _ in pbar:
#                 images = images.to(device)
#                 outputs = model(images)
#                 preds = outputs.cpu().numpy()
#                 all_preds.append(preds)
        
#         all_preds = np.concatenate(all_preds)
        
#         # 创建提交DataFrame
#         submission_df = pd.DataFrame(
#             all_preds, 
#             columns=['Aneurysm Present'] + location_labels
#         )
#         submission_df.insert(0, 'SeriesInstanceUID', all_ids)
        
#         return submission_df
    
#     # 生成并保存提交文件
#     submission_df = generate_submission(model, test_loader, test_dataset, device)
#     submission_df.to_parquet('submission.parquet', index=False, compression='snappy')
    
#     # 验证提交文件
#     if os.path.exists('submission.parquet'):
#         file_size = os.path.getsize('submission.parquet') / 1024 / 1024
#         print(f"提交文件已保存: submission.parquet ({file_size:.2f} MB)")
#         print("前5行预览:")
#         print(submission_df.head())
#     else:
#         print("错误：提交文件未生成")
    
#     print("\n===== 所有流程执行完成 =====")

# if __name__ == "__main__":
#     main()



# 动脉瘤检测完整代码（无函数嵌套版）
# 所有模块合并为单一文件，保留独立性，可按顺序执行

# ==================================================
# 1. 导入依赖库与基础配置
# ==================================================
import os
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTConfig
from safetensors.torch import load_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import logging
import json
from pydicom.pixel_data_handlers.util import apply_voi_lut

# 配置日志记录
logging.basicConfig(
    filename='dicom_processing.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用设备: {device}")

# 定义动脉瘤位置标签
location_labels = [
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
    'Other Posterior Circulation'
]
all_label_cols = ['Aneurysm Present'] + location_labels
print(f"标签列总数: {len(all_label_cols)} (1个存在标签 + 13个位置标签)")


# ==================================================
# 2. 数据加载与预处理
# ==================================================
print("\n===== 开始数据加载与清洗 =====")
# 加载原始数据（Kaggle数据集路径示例）
train_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
localizers_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')

print(f"原始训练数据形状: {train_df.shape}")
print(f"定位器数据形状: {localizers_df.shape}")

# 处理缺失值
print(f"\n预处理前缺失值总数: {train_df.isnull().sum().sum()}")
train_df.fillna(0, inplace=True)  # 标签缺失填充为0（无动脉瘤）
localizers_df.dropna(inplace=True)  # 定位器数据直接删除缺失行
print(f"预处理后缺失值总数: {train_df.isnull().sum().sum()}")

# 统一标签类型
for col in all_label_cols:
    if col in train_df.columns:
        if train_df[col].dtype == 'object':
            train_df[col] = train_df[col].apply(lambda x: 
                1.0 if str(x).strip().lower() in ['true', 'yes', 'present', '1'] else
                0.0 if str(x).strip().lower() in ['false', 'no', 'absent', '0'] else
                float(x) if pd.notna(x) and str(x).strip() else 0.0
            )
        train_df[col] = train_df[col].astype(np.float32)

# 数据集划分（7:1:2）
print("\n===== 开始数据集划分 =====")
train_df_split, test_df = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df['Aneurysm Present']
)
train_df_final, val_df = train_test_split(
    train_df_split, 
    test_size=0.125, 
    random_state=42, 
    stratify=train_df_split['Aneurysm Present']
)

# 输出划分结果
print(f"训练集大小: {len(train_df_final)}")
print(f"验证集大小: {len(val_df)}")
print(f"测试集大小: {len(test_df)}")
print("\n各数据集动脉瘤存在比例:")
print(f"训练集: {train_df_final['Aneurysm Present'].mean():.4f}")
print(f"验证集: {val_df['Aneurysm Present'].mean():.4f}")
print(f"测试集: {test_df['Aneurysm Present'].mean():.4f}")

# 保存清洗后的数据集
train_df_final.to_csv('cleaned_train.csv', index=False)
val_df.to_csv('cleaned_val.csv', index=False)
test_df.to_csv('cleaned_test.csv', index=False)
print("\n清洗后的数据集已保存为: cleaned_train.csv / cleaned_val.csv / cleaned_test.csv")


# ==================================================
# 3. 自定义数据集类实现
# ==================================================
# 重新加载清洗后的数据集
train_df = pd.read_csv('cleaned_train.csv')
val_df = pd.read_csv('cleaned_val.csv')
test_df = pd.read_csv('cleaned_test.csv')

# 定义数据集类
class AneurysmDataset(Dataset):
    def __init__(self, df, root_dir='/kaggle/input/rsna-intracranial-aneurysm-detection/series', 
                 transform=None, is_test=False, location_labels=None):
        self.df = df.copy()
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.location_labels = location_labels
        self.target_shape = (1, 224, 224)
        self.label_shape = (14,)
        
        self._validate_and_clean_labels()
        
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"DICOM根目录不存在: {self.root_dir}")
            
        valid_indices = self._filter_valid_series()
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"有效样本数（过滤后）: {len(self.df)}")

    def _validate_and_clean_labels(self):
        if not self.location_labels:
            raise ValueError("location_labels 不能为空，请传入位置标签列表")
            
        all_label_cols = ['Aneurysm Present'] + self.location_labels
        missing_cols = [col for col in all_label_cols if col not in self.df.columns]
        if missing_cols:
            raise KeyError(f"数据框缺少标签列: {missing_cols}")
        
        for col in all_label_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col].fillna(0.0, inplace=True)
            self.df[col] = self.df[col].astype(np.float32)

    def _filter_valid_series(self):
        valid_indices = []
        for idx in tqdm(range(len(self.df)), desc="过滤无效样本"):
            series_id = self.df.iloc[idx]['SeriesInstanceUID']
            series_path = os.path.join(self.root_dir, series_id)
            if os.path.exists(series_path) and len(os.listdir(series_path)) > 0:
                valid_indices.append(idx)
            else:
                warn_msg = f"样本{idx}路径不存在或为空: {series_path}"
                logging.warning(warn_msg)
        return valid_indices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        series_id = self.df.iloc[idx]['SeriesInstanceUID']
        series_path = os.path.join(self.root_dir, series_id)
        
        image = torch.zeros(self.target_shape, dtype=torch.float32)
        label = torch.zeros(self.label_shape, dtype=torch.float32)

        try:
            dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
            if not dicom_files:
                warn_msg = f"样本{idx}未找到DICOM文件"
                logging.warning(warn_msg)
                return image, label
            
            dicom_files.sort(key=lambda x: int(x.split('.')[0]))
            mid_idx = len(dicom_files) // 2
            dicom_path = os.path.join(series_path, dicom_files[mid_idx])
            
            dicom = pydicom.dcmread(dicom_path)
            pixel_array = apply_voi_lut(dicom.pixel_array, dicom)
            
            if len(pixel_array.shape) == 3:
                slice_idx = pixel_array.shape[0] // 2
                pixel_array = pixel_array[slice_idx, :, :]
                logging.warning(f"样本{idx}是3D数据，已取中间切片")
            elif len(pixel_array.shape) != 2:
                error_msg = f"样本{idx}DICOM维度异常: {pixel_array.shape}"
                logging.error(error_msg)
                return image, label
            
            pixel_array = pixel_array.astype(np.float32)
            if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
                pixel_array = pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept
            
            img_min, img_max = pixel_array.min(), pixel_array.max()
            if img_max != img_min:
                pixel_array = (pixel_array - img_min) / (img_max - img_min)
            
            img_tensor = torch.from_numpy(pixel_array).unsqueeze(0)
            resize_transform = transforms.Resize((self.target_shape[1], self.target_shape[2]))
            img_tensor = resize_transform(img_tensor)
            
            if img_tensor.shape != self.target_shape:
                raise RuntimeError(f"样本{idx}图像形状错误: {img_tensor.shape}")
            
            image = img_tensor
            
        except Exception as e:
            error_msg = f"样本{idx}处理DICOM时出错: {str(e)}"
            logging.error(error_msg)
        
        try:
            if not self.is_test:
                present = float(self.df.iloc[idx]['Aneurysm Present'])
                locations = self.df.iloc[idx][self.location_labels].values.astype(np.float32)
                label_np = np.concatenate([[present], locations])
                label = torch.FloatTensor(label_np)
        except Exception as e:
            error_msg = f"样本{idx}处理标签时出错: {str(e)}"
            logging.error(error_msg)
        
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                logging.error(f"样本{idx}应用变换时出错: {str(e)}")
        
        return image, label

# 测试数据集加载
print("\n===== 测试数据集加载 =====")
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 创建数据集实例
train_dataset = AneurysmDataset(
    train_df, 
    transform=train_transform, 
    location_labels=location_labels
)
val_dataset = AneurysmDataset(
    val_df, 
    transform=val_test_transform, 
    location_labels=location_labels
)
test_dataset = AneurysmDataset(
    test_df, 
    transform=val_test_transform, 
    is_test=False,
    location_labels=location_labels
)

# 输出数据集信息
print(f"\n训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

# 测试单样本加载
sample_img, sample_label = train_dataset[0]
print(f"\n单样本图像形状: {sample_img.shape}")
print(f"单样本标签形状: {sample_label.shape}")
print(f"单样本标签值: {sample_label.numpy()}")


# ==================================================
# 4. 模型定义与初始化
# ==================================================
class AneurysmViT(nn.Module):
    def __init__(self, num_labels=14):
        super(AneurysmViT, self).__init__()
        
        self.vit_config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            dropout=0.3,
            attention_probs_dropout_prob=0.3
        )
        
        self.vit = ViTModel(self.vit_config)
        
        model_path = '/kaggle/input/vit/transformers/default/1/vit_model/model.safetensors'
        if os.path.exists(model_path):
            print(f"正在加载预训练模型: {model_path}")
            state_dict = load_file(model_path)
            
            cleaned_state_dict = {}
            for param_name, param_value in state_dict.items():
                if param_name.startswith('vit.'):
                    cleaned_name = param_name[4:]
                else:
                    cleaned_name = param_name
                cleaned_state_dict[cleaned_name] = param_value
            
            self.vit.load_state_dict(cleaned_state_dict, strict=True)
            print("预训练权重加载成功")
        else:
            raise FileNotFoundError(f"预训练模型文件不存在: {model_path}")
        
        # 冻结部分预训练层
        freeze_layers = 8
        for i, (name, param) in enumerate(self.vit.named_parameters()):
            if i < freeze_layers * 12:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )
        
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # 1通道转3通道
        vit_outputs = self.vit(pixel_values=x)
        cls_token = vit_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return torch.sigmoid(logits)

# 初始化模型并验证
print("\n===== 模型初始化与结构验证 =====")
model = AneurysmViT(num_labels=14).to(device)
print(f"模型已移动到 {device} 设备")

# 打印模型结构概览
print("\n模型结构概览:")
print(model)

# 计算模型参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n模型参数统计:")
print(f"总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")
print(f"冻结参数数量: {total_params - trainable_params:,}")

# 测试模型前向传播
test_input = torch.randn(2, 1, 224, 224).to(device)
test_output = model(test_input)
print(f"\n前向传播测试:")
print(f"输入形状: {test_input.shape}")
print(f"输出形状: {test_output.shape}")
print(f"输出值范围: [{test_output.min():.4f}, {test_output.max():.4f}]")


# ==================================================
# 5. 数据加载器创建
# ==================================================
print("\n===== 重新创建Dataset实例 =====")
train_dataset = AneurysmDataset(
    train_df,
    root_dir='/kaggle/input/rsna-intracranial-aneurysm-detection/series',
    transform=train_transform,
    location_labels=location_labels
)
val_dataset = AneurysmDataset(
    val_df,
    root_dir='/kaggle/input/rsna-intracranial-aneurysm-detection/series',
    transform=val_test_transform,
    location_labels=location_labels
)
test_dataset = AneurysmDataset(
    test_df,
    root_dir='/kaggle/input/rsna-intracranial-aneurysm-detection/series',
    transform=val_test_transform,
    is_test=False,
    location_labels=location_labels
)

# 创建DataLoader
print("\n===== 创建DataLoader =====")
batch_size = 16
num_workers = 2 if device.type == 'cuda' else 0
pin_memory = True if device.type == 'cuda' else False

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False
)

# 验证Loader有效性
print(f"\nDataLoader验证:")
print(f"训练集Loader批次数: {len(train_loader)} (每批{batch_size}样本)")
print(f"验证集Loader批次数: {len(val_loader)}")
print(f"测试集Loader批次数: {len(test_loader)}")

# 测试批量加载数据
train_batch = next(iter(train_loader))
val_batch = next(iter(val_loader))
print(f"\n训练集批量数据形状:")
print(f"图像形状: {train_batch[0].shape}")
print(f"标签形状: {train_batch[1].shape}")
print(f"验证集批量数据形状:")
print(f"图像形状: {val_batch[0].shape}")
print(f"标签形状: {val_batch[1].shape}")

# 保存Loader配置
loader_config = {
    'batch_size': batch_size,
    'num_workers': num_workers,
    'pin_memory': pin_memory,
    'train_loader_len': len(train_loader),
    'val_loader_len': len(val_loader),
    'test_loader_len': len(test_loader)
}
with open('loader_config.json', 'w') as f:
    json.dump(loader_config, f, indent=2)
print("\nLoader配置已保存为: loader_config.json")


# ==================================================
# 6. 训练配置与模型训练
# ==================================================
# 计算类别权重
print("\n===== 计算类别权重 =====")
present_count = train_df['Aneurysm Present'].sum()
total_count = len(train_df)
weight_positive = total_count / (2 * present_count) if present_count > 0 else 1.0
print(f"训练集动脉瘤阳性样本占比: {present_count/total_count:.4f}")
print(f"阳性样本权重: {weight_positive:.2f}")

# 定义加权BCE损失
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, output, target):
        weights = torch.ones_like(target, device=device)
        weights[:, 0] = self.pos_weight
        bce_loss = nn.BCELoss(reduction='none')(output, target)
        return (bce_loss * weights).mean()

# 配置优化器与调度器
print("\n===== 配置优化器与调度器 =====")
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,
    T_mult=2,
    eta_min=1e-6
)

# 训练参数初始化
num_epochs = 15
best_val_score = 0.0
early_stop_patience = 5
early_stop_counter = 0

# 记录训练指标
train_losses = []
val_losses = []
train_aucs = []
val_aucs = []

# 开始训练
print("\n===== 开始模型训练 =====")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_auc = 0.0
    train_auc_count = 0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 训练")
    for images, labels in train_pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = WeightedBCELoss(pos_weight=weight_positive)(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        
        # 计算AUC
        labels_np = labels.cpu().detach().numpy()
        outputs_np = outputs.cpu().detach().numpy()
        if len(np.unique(labels_np[:, 0])) >= 2:
            auc = roc_auc_score(labels_np[:, 0], outputs_np[:, 0])
            train_auc += auc * images.size(0)
            train_auc_count += images.size(0)
        
        train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    
    # 计算训练集平均指标
    train_loss_avg = train_loss / len(train_loader.dataset)
    train_auc_avg = train_auc / train_auc_count if train_auc_count > 0 else 0.5
    train_losses.append(train_loss_avg)
    train_aucs.append(train_auc_avg)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_scores = []
    val_main_aucs = []
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 验证")
        for images, labels in val_pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = WeightedBCELoss(pos_weight=weight_positive)(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            # 计算各标签AUC
            labels_np = labels.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            auc_scores = []
            
            for i in range(labels_np.shape[1]):
                try:
                    if len(np.unique(labels_np[:, i])) >= 2:
                        auc_scores.append(roc_auc_score(labels_np[:, i], outputs_np[:, i]))
                    else:
                        auc_scores.append(0.5)
                except:
                    auc_scores.append(0.5)
            
            main_auc = auc_scores[0]
            other_auc = np.mean(auc_scores[1:])
            val_scores.append(0.5 * (main_auc + other_auc))
            val_main_aucs.append(main_auc)
            
            val_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    
    # 计算验证集平均指标
    val_loss_avg = val_loss / len(val_loader.dataset)
    val_score_avg = np.mean(val_scores)
    val_main_auc_avg = np.mean(val_main_aucs)
    val_losses.append(val_loss_avg)
    val_aucs.append(val_main_auc_avg)
    
    # 更新调度器
    scheduler.step()
    
    # 保存最佳模型
    if val_score_avg > best_val_score:
        best_val_score = val_score_avg
        torch.save(model.state_dict(), 'best_model.pth')
        early_stop_counter = 0
        print(f"✅ 保存最佳模型（验证综合评分: {val_score_avg:.4f}）")
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print(f"❌ 早停触发：连续{early_stop_patience}轮验证分数无提升")
            break
    
    # 打印本轮结果
    print(f"\nEpoch {epoch+1}/{num_epochs} 结果:")
    print(f"训练损失: {train_loss_avg:.4f} | 训练主标签AUC: {train_auc_avg:.4f}")
    print(f"验证损失: {val_loss_avg:.4f} | 验证主标签AUC: {val_main_auc_avg:.4f} | 验证综合评分: {val_score_avg:.4f}")
    print("-" * 80)

print("训练完成!")

# 保存训练指标
np.savez('training_metrics.npz', 
         train_losses=train_losses, 
         val_losses=val_losses,
         train_aucs=train_aucs,
         val_aucs=val_aucs)
print("训练指标已保存为: training_metrics.npz")


# ==================================================
# 7. 训练历史可视化
# ==================================================
print("\n===== 生成训练历史可视化 =====")
try:
    metrics = np.load('training_metrics.npz')
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    train_aucs = metrics['train_aucs']
    val_aucs = metrics['val_aucs']
    num_epochs = len(train_losses)
    print(f"成功加载训练指标，共{num_epochs}个训练轮次")
except FileNotFoundError:
    print("错误：未找到training_metrics.npz，请先运行训练模块")

# 创建可视化图表
plt.figure(figsize=(14, 6))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss', marker='o', markersize=4)
plt.plot(range(1, num_epochs+1), val_losses, label='Validation loss', marker='s', markersize=4)
plt.title('Training and Validation Loss Curve', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Loss Value', fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# AUC曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_aucs, label='TrainingAUC', marker='o', markersize=4)
plt.plot(range(1, num_epochs+1), val_aucs, label='VerificationAUC', marker='s', markersize=4)
plt.title('Training and Validation AUC Curve (Primary Label)', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('AUC value', fontsize=10)
plt.ylim(0.5, 1.0)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# 保存与显示
plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("训练历史图表已保存为: training_history.png")
plt.show()


# ==================================================
# 8. 测试集评估
# ==================================================
print("\n===== 开始测试集评估 =====")
# 重新创建测试集Loader
test_dataset = AneurysmDataset(
    test_df,
    root_dir='/kaggle/input/rsna-intracranial-aneurysm-detection/series',
    transform=val_test_transform,
    is_test=False,
    location_labels=location_labels
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2 if device.type == 'cuda' else 0,
    pin_memory=True if device.type == 'cuda' else False
)

# 加载最佳模型
model = AneurysmViT(num_labels=14).to(device)
try:
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    print("成功加载最佳模型: best_model.pth")
except FileNotFoundError:
    print("错误：未找到best_model.pth，请先运行训练模块")

model.eval()

# 测试集评估
test_scores = []
test_aucs = []
all_auc_scores = []

with torch.no_grad():
    test_pbar = tqdm(test_loader, desc="测试集评估")
    for images, labels in test_pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        # 计算各标签AUC
        labels_np = labels.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
        auc_scores = []
        
        for i in range(labels_np.shape[1]):
            try:
                auc = roc_auc_score(labels_np[:, i], outputs_np[:, i])
                auc_scores.append(auc)
            except ValueError:
                auc_scores.append(0.5)
        
        all_auc_scores.append(auc_scores)
        
        # 计算综合评分
        main_auc = auc_scores[0]
        other_auc = np.mean(auc_scores[1:])
        weighted_score = 0.5 * (main_auc + other_auc)
        
        test_scores.append(weighted_score)
        test_aucs.append(main_auc)

# 计算最终指标
test_score = np.mean(test_scores)
test_auc = np.mean(test_aucs)
mean_auc_per_label = np.mean(all_auc_scores, axis=0)

# 输出评估结果
print(f"\n测试集最终结果:")
print(f"主标签（是否存在动脉瘤）AUC: {test_auc:.4f}")
print(f"综合评分（主标签+位置标签）: {test_score:.4f}")

print("\n各标签AUC值:")
for i, label in enumerate(['Aneurysm Present'] + location_labels):
    print(f"{label}: {mean_auc_per_label[i]:.4f}")

# 保存评估结果
eval_results = {
    'main_auc': test_auc,
    'overall_score': test_score,
    'per_label_auc': dict(zip(['Aneurysm Present'] + location_labels, mean_auc_per_label))
}
with open('evaluation_results.json', 'w') as f:
    json.dump(eval_results, f, indent=2)
print("\n评估结果已保存为: evaluation_results.json")


# ==================================================
# 9. 预测结果可视化
# ==================================================
print("\n===== 生成预测结果可视化 =====")
num_samples = 5
indices = np.random.choice(len(test_dataset), num_samples, replace=False)

plt.figure(figsize=(15, 4*num_samples))
for i, idx in enumerate(indices):
    image, label = test_dataset[idx]
    image_tensor = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(image_tensor).cpu().numpy()[0]
    
    # 显示医学图像
    plt.subplot(num_samples, 2, 2*i+1)
    plt.imshow(image[0], cmap='gray')
    plt.title(f"样本 {idx}\n true value: {label[0].item():.0f} | Predicted Value: {pred[0]:.2f}", fontsize=10)
    plt.axis('off')
    
    # 显示位置预测概率
    plt.subplot(num_samples, 2, 2*i+2)
    plt.barh(location_labels, pred[1:])
    plt.xlim(0, 1)
    plt.title("Aneurysm location prediction probability", fontsize=10)
    plt.yticks(fontsize=8)

# 保存与显示
plt.tight_layout()
plt.savefig('predictions_visualization.png', dpi=300, bbox_inches='tight')
print("预测结果可视化已保存为: predictions_visualization.png")
plt.show()


# ==================================================
# 10. 生成提交文件
# ==================================================
print("\n===== 生成提交文件 =====")
# 重新创建测试集Loader（测试模式）
test_dataset = AneurysmDataset(
    test_df,
    root_dir='/kaggle/input/rsna-intracranial-aneurysm-detection/series',
    transform=val_test_transform,
    is_test=True,
    location_labels=location_labels
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2 if device.type == 'cuda' else 0,
    pin_memory=True if device.type == 'cuda' else False
)

# 加载最佳模型
model = AneurysmViT(num_labels=14).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# 生成预测结果
all_preds = []
all_ids = test_dataset.df['SeriesInstanceUID'].values

with torch.no_grad():
    pbar = tqdm(test_loader, desc="生成预测结果")
    for images, _ in pbar:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.cpu().numpy()
        all_preds.append(preds)

# 合并所有预测结果
all_preds = np.concatenate(all_preds)

# 创建提交DataFrame
submission_df = pd.DataFrame(
    all_preds,
    columns=['Aneurysm Present'] + location_labels
)
submission_df.insert(0, 'SeriesInstanceUID', all_ids)

# 保存提交文件
submission_df.to_parquet('submission.parquet', index=False, compression='snappy')

# 验证提交文件
if os.path.exists('submission.parquet'):
    file_size = os.path.getsize('submission.parquet') / 1024 / 1024
    print(f"提交文件已保存: submission.parquet ({file_size:.2f} MB)")
    print("前5行预览:")
    print(submission_df.head())
else:
    print("错误：提交文件未生成")

print("\n===== 所有流程执行完成 =====")



import pandas as pd

# 创建一个简单的DataFrame作为示例数据
data = {
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'value': [10.5, 20.3, 15.7, 25.1, 30.0]
}
df = pd.DataFrame(data)

# 将DataFrame保存为parquet文件
df.to_parquet('submission.parquet', index=False)

print("文件已成功保存为submission.parquet")
    

