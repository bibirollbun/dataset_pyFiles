# 快速查看可用数据集
import os
print(os.listdir("/kaggle/input"))


import os

# 1. 定义输入数据目录（根据实际需求选择数据集）
INPUT_DATA = "/kaggle/input/my-pneumonia-gan"  # 替换为你要使用的数据集

# 2. 验证路径是否正确
print("\n=== 路径验证 ===")
print(f"输入数据目录: {INPUT_DATA}")
print("目录内容:", os.listdir(INPUT_DATA))

# 3. 验证关键文件是否存在
essential_files = ["final_model/model.pth", "train_split.csv"]
for file in essential_files:
    path = f"{INPUT_DATA}/{file}"
    print(f"{'√' if os.path.exists(path) else '×'} {path}")


import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# ===== 1. 定义与训练时完全一致的Generator =====
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.conv(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = 128 // 4  # IMG_SIZE=128
        self.l1 = nn.Sequential(
            spectral_norm(nn.Linear(128, 128 * self.init_size ** 2))  # LATENT_DIM=128
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(128, 64, 3, stride=1, padding=1)),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 1, 3, stride=1, padding=1)),  # CHANNELS=1
            nn.Tanh()
        )
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        return self.conv_blocks(out)

# ===== 2. 加载训练好的模型 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

# 加载检查点（替换为您的实际路径）
checkpoint_path = "/kaggle/input/my-pneumonia-gan/final_model/model.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
generator.load_state_dict(checkpoint['generator'])  # 严格匹配
generator.eval()
print("✅ 模型加载成功")

# ===== 3. 生成增强样本 =====
def generate_synthetic_samples(num_samples_normal, num_samples_pneumonia, output_dir="/kaggle/working/synthetic_images"):
    """生成指定数量的正常和肺炎样本"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成正常样本 (label=0)
    normal_samples = []
    for i in range(num_samples_normal):
        z = torch.randn(1, 128).to(device)  # LATENT_DIM=128
        with torch.no_grad():
            img = generator(z).cpu().squeeze().numpy()
            img = (img * 127.5 + 127.5).astype(np.uint8)  # 转换到0-255范围
            img_path = f"{output_dir}/synth_normal_{i}.png"
            Image.fromarray(img).save(img_path)
            normal_samples.append({
                'patientId': f'synth_normal_{i}',
                'dcm_path': img_path,
                'label': 0
            })
        
        if i % 100 == 0:
            print(f"已生成正常样本 {i+1}/{num_samples_normal}")
    
    # 生成肺炎样本 (label=1)
    pneumonia_samples = []
    for i in range(num_samples_pneumonia):
        z = torch.randn(1, 128).to(device)  # LATENT_DIM=128
        with torch.no_grad():
            img = generator(z).cpu().squeeze().numpy()
            img = (img * 127.5 + 127.5).astype(np.uint8)  # 转换到0-255范围
            img_path = f"{output_dir}/synth_pneumonia_{i}.png"
            Image.fromarray(img).save(img_path)
            pneumonia_samples.append({
                'patientId': f'synth_pneumonia_{i}',
                'dcm_path': img_path,
                'label': 1
            })
        
        if i % 100 == 0:
            print(f"已生成肺炎样本 {i+1}/{num_samples_pneumonia}")
    
    # 合并并返回元数据
    all_samples = normal_samples + pneumonia_samples
    return pd.DataFrame(all_samples)

# ===== 4. 智能计算需要生成的样本数量 =====
def calculate_samples_to_generate(original_data_path="/kaggle/input/my-pneumonia-gan/train_split.csv"):
    """根据原始数据的类别分布，计算需要生成的正常和肺炎样本数量"""
    try:
        original_df = pd.read_csv(original_data_path)
        class_distribution = original_df['label'].value_counts()
        
        # 0=正常，1=肺炎
        normal_count = class_distribution.get(0, 0)
        pneumonia_count = class_distribution.get(1, 0)
        
        print(f"Original dataset distribution: Normal={normal_count}, Pneumonia={pneumonia_count}")
        
        # 计算需要生成的样本数量，目标是使两类数量平衡
        if normal_count > pneumonia_count:
            # Need more pneumonia samples
            return 0, normal_count - pneumonia_count
        else:
            # Need more normal samples
            return pneumonia_count - normal_count, 0
    except Exception as e:
        print(f"Error calculating sample counts: {e}")
        print("Using default values: generating 500 normal and 500 pneumonia samples")
        return 500, 500

# ===== 5. 主函数 =====
if __name__ == "__main__":
    # 计算需要生成的样本数量
    num_normal, num_pneumonia = calculate_samples_to_generate()
    
    # 如果不需要生成任何样本，则使用默认值
    if num_normal == 0 and num_pneumonia == 0:
        print("数据已平衡，无需生成额外样本")
        num_normal, num_pneumonia = 500, 500
    
    print(f"计划生成: 正常={num_normal}, 肺炎={num_pneumonia}")
    
    # 生成样本并获取元数据
    synthetic_metadata = generate_synthetic_samples(num_normal, num_pneumonia)
    
    # 保存元数据
    metadata_path = "/kaggle/working/synthetic_samples_metadata.csv"
    synthetic_metadata.to_csv(metadata_path, index=False)
    print(f"✅ 合成样本元数据已保存到 {metadata_path}")
    
    # ===== 6. 验证生成质量 =====
    print("\n=== 生成样本示例 ===")
    sample_files = [f for f in os.listdir("/kaggle/working/synthetic_images") if f.endswith('.png')][:4]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for i, file in enumerate(sample_files):
        img = Image.open(f"/kaggle/working/synthetic_images/{file}")
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title('Normal' if 'normal' in file else 'Pneumonia')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('/kaggle/working/synthetic_samples_example.jpg')
    plt.show()
    
    # ===== 7. 分析生成样本的类别分布 =====
    plt.figure(figsize=(8, 5))
    class_counts = synthetic_metadata['label'].value_counts()
    # 确保标签映射正确（0=Normal，1=Pneumonia）
    class_labels = {0: "Normal", 1: "Pneumonia"}
    class_counts.index = [class_labels[idx] for idx in class_counts.index]  # 重命名索引

    class_counts.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Synthetic Sample Class Distribution')  # 英文标题
    plt.xticks(rotation=0)  # 刻度旋转（可选，保持水平）
    plt.xlabel('Class')      # 新增X轴标签
    plt.ylabel('Number of Samples')  # 英文Y轴标签
    plt.savefig('/kaggle/working/synthetic_class_distribution.jpg', bbox_inches='tight')  # 优化保存
    plt.show()


# ===== 1. 导入所有依赖 =====
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# ===== 2. 验证生成图像质量 =====
# 创建输出目录（如果不存在）
os.makedirs("/kaggle/working/synthetic_images", exist_ok=True)

# 随机检查10张生成图像
sample_files = random.sample(os.listdir("/kaggle/working/synthetic_images"), min(10, len(os.listdir("/kaggle/working/synthetic_images"))))

plt.figure(figsize=(15, 5))
for i, file in enumerate(sample_files):
    img = Image.open(f"/kaggle/working/synthetic_images/{file}")
    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
# ===== 3. 构建增强数据集（修正版） =====
try:
    # 加载原始训练集
    train_df = pd.read_csv("/kaggle/input/my-pneumonia-gan/train_split.csv")
    
    # 加载合成样本的元数据（包含正确标签）
    synth_metadata_path = "/kaggle/working/synthetic_samples_metadata.csv"
    if not os.path.exists(synth_metadata_path):
        raise FileNotFoundError(f"合成样本元数据文件不存在: {synth_metadata_path}")
    
    synth_df = pd.read_csv(synth_metadata_path)
    
    # 合并数据集（保持元数据中的正确标签）
    augmented_df = pd.concat([train_df, synth_df], ignore_index=True)
    augmented_df.to_csv("/kaggle/working/augmented_train.csv", index=False)
    
    print(f"增强后数据集：原始 {len(train_df)} + 合成 {len(synth_df)} = {len(augmented_df)} 张")
    print("前5条记录：\n", augmented_df.head())
    
    # 验证类别分布
    class_distribution = augmented_df['label'].value_counts()
    print("\n增强后类别分布:")
    print(f"正常样本 (label=0): {class_distribution.get(0, 0)}")
    print(f"肺炎样本 (label=1): {class_distribution.get(1, 0)}")
    
except Exception as e:
    print(f"发生错误：{str(e)}")
    print("请检查：")
    print("1. 原始CSV文件路径是否正确")
    print("2. 合成样本元数据是否已生成")
    print("3. 元数据文件路径是否正确")


import matplotlib.pyplot as plt
import pandas as pd

# 1. 加载增强数据集
augmented_df = pd.read_csv("/kaggle/working/augmented_train.csv")

# 2. 计算类别分布（这一步必须在使用class_dist之前！）
class_dist = augmented_df['label'].value_counts()

# 3. 显式映射标签含义（避免硬编码）
label_mapping = {0: 'Normal', 1: 'Pneumonia'}
class_dist.index = [label_mapping[idx] for idx in class_dist.index]  # 现在class_dist已定义

# 4. 可视化（英文标题）
plt.figure(figsize=(8, 4))
plt.pie(
    class_dist, 
    labels=class_dist.index,  # 使用映射后的标签
    autopct='%1.1f%%', 
    colors=['lightgreen', 'lightcoral'],
    textprops={'fontsize': 12}
)  
plt.title("Class Distribution in Augmented Dataset", fontsize=14, pad=20)
plt.savefig('/kaggle/working/class_balance.jpg', dpi=300, bbox_inches='tight')
plt.show()


# 检查当前增强数据集的真实/合成样本比例及类别分布
real_samples = augmented_df[~augmented_df['dcm_path'].str.contains('synthetic')]
synth_samples = augmented_df[augmented_df['dcm_path'].str.contains('synthetic')]

# 复用标签映射（与可视化一致：0=Normal，1=Pneumonia）
label_mapping = {0: 'Normal', 1: 'Pneumonia'}

# ========== 真实样本统计 ==========
real_class_dist = real_samples['label'].value_counts().rename(index=label_mapping)
real_pneumonia_ratio = real_samples['label'].mean()  # label=1的比例

print(f"真实样本: {len(real_samples)}")
for cls in label_mapping.values():
    count = real_class_dist.get(cls, 0)
    print(f"  - {cls}: {count}")
print(f"  - 肺炎占比: {real_pneumonia_ratio:.1%}")

# ========== 合成样本统计 ==========
synth_class_dist = synth_samples['label'].value_counts().rename(index=label_mapping)

print(f"\n合成样本: {len(synth_samples)}")
for cls in label_mapping.values():
    count = synth_class_dist.get(cls, 0)
    print(f"  - {cls}: {count}")

# ========== 整体肺炎占比 ==========
overall_pneumonia_ratio = augmented_df['label'].mean()
print(f"\n肺炎总占比: {overall_pneumonia_ratio:.1%}")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import pydicom  # 添加DICOM处理库
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# ====================== 1. 数据集类 ======================
class PneumoniaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['dcm_path']
        
        # 检查文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"错误：在{img_path}处未找到图像")
        
        # 处理DICOM文件
        if img_path.lower().endswith('.dcm'):
            try:
                dicom = pydicom.dcmread(img_path)
                pixel_array = dicom.pixel_array
                
                # 将像素数组转换为PIL图像
                if pixel_array.dtype != np.uint8:
                    # 转换为8位图像（如果需要）
                    pixel_array = self._convert_to_uint8(pixel_array)
                    
                image = Image.fromarray(pixel_array)
                image = image.convert('RGB')  # 确保为RGB格式
            except Exception as e:
                raise RuntimeError(f"无法处理DICOM文件 {img_path}: {str(e)}")
        else:
            # 处理普通图像（如PNG/JPG）
            image = Image.open(img_path).convert('RGB')

        label = torch.tensor(row['label'], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def _convert_to_uint8(self, array):
        """将不同位深度的像素数组转换为uint8格式"""
        if np.issubdtype(array.dtype, np.floating):
            # 浮点型数据（如[-1,1]或[0,1]）
            array = (array * 255).astype(np.uint8)
        else:
            # 整型数据（如16位）
            array_min, array_max = array.min(), array.max()
            if array_min == array_max:
                # 避免除以零
                array = np.zeros_like(array, dtype=np.uint8)
            else:
                array = ((array - array_min) / (array_max - array_min) * 255).astype(np.uint8)
        return array


# ====================== 2. 数据准备 ======================
def prepare_data(data_df, batch_size=32, is_augmented=False):
    transforms_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    # 仅对增强模型的训练集添加额外增强
    if is_augmented:
        transforms_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
        transforms_list.insert(2, transforms.RandomRotation(15))
        transforms_list.insert(3, transforms.ColorJitter(brightness=0.2, contrast=0.2))
    
    transform = transforms.Compose(transforms_list)
    dataset = PneumoniaDataset(data_df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader

# ====================== 3. 验证数据加载器 ======================
def validate_data_loader(data_loader, num_samples=5, device="cpu"):
    print(f"\n{'='*50}")
    print(f"验证数据加载器（加载{num_samples}个批次）...")
    for i, (images, labels) in enumerate(data_loader):
        if i >= num_samples:
            break

        images, labels = images.to(device), labels.to(device)

        print(f"批次 {i+1}/{num_samples}：")
        print(f"  图像形状: {images.shape}")
        print(f"  标签分布: {np.bincount(labels.cpu().numpy())}")

        plt.figure(figsize=(4, 4))
        img = images[0].permute(1, 2, 0).cpu().numpy()
        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"label: {labels[0].item()}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    print(f"数据加载器验证完成{'='*50}\n")


# ====================== 4. CNN模型 ======================
class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ====================== 5. 训练函数 ======================
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    best_val_auc = 0.0  # 改用AUC作为保存依据
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    
    # 学习率调度：余弦退火
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        val_acc = 100. * val_correct / val_total
        val_auc = roc_auc_score(all_labels, all_probs)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        print(f'轮次 {epoch+1}/{epochs}')
        print(f'训练损失: {history["train_loss"][-1]:.4f} | 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {history["val_loss"][-1]:.4f} | 验证准确率: {val_acc:.2f}% | 验证AUC: {val_auc:.4f}')

        # 保存最佳模型（基于AUC）
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), '/kaggle/working/best_model_base.pth')
            print(f'保存最佳模型，AUC: {best_val_auc:.4f}')

        # 更新学习率
        scheduler.step()

    return history, best_val_auc

# ====================== 6. 评估函数 ======================
def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['正常', '肺炎'])

    print(f'测试准确率: {acc:.4f}')
    print(f'测试AUC: {auc:.4f}')
    print(f'混淆矩阵:\n{cm}')
    print(f'分类报告:\n{report}')

    return acc, auc, cm, report


# ====================== 7. 主函数（基础模型） ======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_df = pd.read_csv("/kaggle/input/my-pneumonia-gan/train_split.csv")

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])
    print(f"原始训练集大小: {len(train_df)} | 验证集大小: {len(val_df)}")

    train_loader = prepare_data(train_df)
    val_loader = prepare_data(val_df)

    validate_data_loader(train_loader, num_samples=3, device=device)

    model = PneumoniaCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("开始训练基础模型...")
    history, best_acc = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss History')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy History')
    plt.savefig('/kaggle/working/base_model_training_history.jpg')
    plt.show()

    test_df = pd.read_csv("/kaggle/input/my-pneumonia-gan/val_split.csv")
    test_loader = prepare_data(test_df)

    model.load_state_dict(torch.load('/kaggle/working/best_model_base.pth'))
    print("\n评估基础模型性能:")
    acc, auc, cm, report = evaluate_model(model, test_loader, device)

    with open('/kaggle/working/base_model_results.txt', 'w') as f:
        f.write(f'基础模型测试准确率: {acc:.4f}\n')
        f.write(f'基础模型测试AUC: {auc:.4f}\n')
        f.write(f'混淆矩阵:\n{cm}\n')
        f.write(f'分类报告:\n{report}')

    return acc, history


# ====================== 8. 使用增强数据训练（可选） ======================
def train_with_augmented_data(base_accuracy, base_history):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 检查是否存在增强数据
    augmented_csv = "/kaggle/working/augmented_train.csv"
    if not os.path.exists(augmented_csv):
        print(f"\n警告: {augmented_csv} 未找到，跳过增强数据训练。")
        print("请先运行「构建增强数据集」代码，或检查文件路径。")
        return None, None

    augmented_df = pd.read_csv(augmented_csv)

    train_df, val_df = train_test_split(augmented_df, test_size=0.2, random_state=42, stratify=augmented_df['label'])
    print(f"增强训练集大小: {len(train_df)} | 验证集大小: {len(val_df)}")

    train_loader = prepare_data(train_df, is_augmented=True)  # 增强训练集
    val_loader = prepare_data(val_df)  # 验证集不变

    validate_data_loader(train_loader, num_samples=3, device=device)

    model = PneumoniaCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2正则化
    print("开始训练增强模型...")
    
    # 保存增强模型到不同路径
    history, best_acc = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)
    torch.save(model.state_dict(), '/kaggle/working/best_model_augmented.pth')  # 保存增强模型

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss History')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy History')
    plt.savefig('/kaggle/working/augmented_model_training_history.jpg')
    plt.show()

    test_df = pd.read_csv("/kaggle/input/my-pneumonia-gan/val_split.csv")
    test_loader = prepare_data(test_df)

    # 加载增强模型
    model.load_state_dict(torch.load('/kaggle/working/best_model_augmented.pth'))
    print("\n评估增强模型性能:")
    acc, auc, cm, report = evaluate_model(model, test_loader, device)

    with open('/kaggle/working/augmented_model_results.txt', 'w') as f:
        f.write(f'增强模型测试准确率: {acc:.4f}\n')
        f.write(f'增强模型测试AUC: {auc:.4f}\n')
        f.write(f'混淆矩阵:\n{cm}\n')
        f.write(f'分类报告:\n{report}')

    improvement = (acc - base_accuracy) / base_accuracy * 100
    print(f"\n模型准确率提升: {improvement:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.bar(['original', 'enhance'], [base_accuracy, acc], color=['lightblue', 'lightgreen'])
    plt.ylim(0.8, 1.0)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate([base_accuracy, acc]):
        plt.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=12)

    plt.savefig('/kaggle/working/model_comparison.jpg')
    plt.show()

    return acc, history


# ====================== 9. 分析结果 ======================
def analyze_results(base_accuracy, base_history, augmented_accuracy, augmented_history):
    if augmented_accuracy is None:
        print("未训练增强模型，跳过性能比较。")
        return
    
    improvement = (augmented_accuracy - base_accuracy) / base_accuracy * 100
    result_summary = f"""
    ================= 模型性能比较总结 =================
    基础模型准确率: {base_accuracy:.4f}
    增强模型准确率: {augmented_accuracy:.4f}
    准确率提升: {improvement:.2f}%
    
    是否达到5%的提升目标? {'✅ 是' if improvement >= 5 else '❌ 否'}
    =================================================
    """

    print(result_summary)

    with open('/kaggle/working/results_summary.txt', 'w') as f:
        f.write(result_summary)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(base_history['val_acc'], label='Original Val Acc')
    plt.plot(augmented_history['val_acc'], label='Augmented Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(base_history['val_loss'], label='Original Val Loss')
    plt.plot(augmented_history['val_loss'], label='Augmented Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('/kaggle/working/combined_training_history.jpg')
    plt.show()

    return result_summary


# ====================== 主执行 ======================
if __name__ == "__main__":
    try:
        # 确保pydicom库已安装
        import pydicom
    except ImportError:
        print("错误：缺少pydicom库。请安装：")
        print("!pip install pydicom")
        exit()
        
    try:
        base_accuracy, base_history = main()
        
        # 若基础模型训练成功，尝试增强训练
        if base_accuracy is not None:
            augmented_accuracy, augmented_history = train_with_augmented_data(base_accuracy, base_history)
            analyze_results(base_accuracy, base_history, augmented_accuracy, augmented_history)
                
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("建议检查：")
        print("1. 数据集文件是否存在（train_split.csv、val_split.csv、augmented_train.csv）")
        print("2. 图像路径是否正确（dcm_path字段是否有效）")
        print("3. 环境依赖是否完整（PyTorch、Torchvision、pydicom等）")

