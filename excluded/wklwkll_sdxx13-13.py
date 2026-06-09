# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install py7zr


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
import py7zr  # 用于解压7z文件

print("所有必要的库已导入")


# 设置随机种子保证可复现性
torch.manual_seed(42)
np.random.seed(42)

# Kaggle数据集路径
KAGGLE_DATA_DIR = "/kaggle/input/cifar-10"
WORKING_DIR = "/kaggle/working"

print(f"数据集目录: {KAGGLE_DATA_DIR}")
print(f"工作目录: {WORKING_DIR}")


def extract_data():
    # 确保工作目录存在
    os.makedirs(os.path.join(WORKING_DIR, "data"), exist_ok=True)
    
    # 检查数据是否已解压
    train_dir = os.path.join(WORKING_DIR, "data/train")
    test_dir = os.path.join(WORKING_DIR, "data/test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("开始解压数据集...")
        
        # 解压训练集
        train_7z_path = os.path.join(KAGGLE_DATA_DIR, "train.7z")
        print(f"解压训练集: {train_7z_path}")
        with py7zr.SevenZipFile(train_7z_path, mode='r') as z:
            z.extractall(path=os.path.join(WORKING_DIR, "data"))
        print("训练集解压完成")
        
        # 解压测试集
        test_7z_path = os.path.join(KAGGLE_DATA_DIR, "test.7z")
        print(f"解压测试集: {test_7z_path}")
        with py7zr.SevenZipFile(test_7z_path, mode='r') as z:
            z.extractall(path=os.path.join(WORKING_DIR, "data"))
        print("测试集解压完成")
        
        # 重命名解压后的文件夹
        os.rename(os.path.join(WORKING_DIR, "data/train"), train_dir)
        os.rename(os.path.join(WORKING_DIR, "data/test"), test_dir)
        print(f"数据已保存到: {WORKING_DIR}/data")
    else:
        print("数据集已存在，跳过解压")
    
    return train_dir, test_dir

# 执行解压
train_dir, test_dir = extract_data()
print(f"训练集目录: {train_dir}")
print(f"测试集目录: {test_dir}")


def prepare_data():
    # 加载训练标签
    labels_path = os.path.join(KAGGLE_DATA_DIR, "trainLabels.csv")
    print(f"加载标签文件: {labels_path}")
    labels_df = pd.read_csv(labels_path)
    
    # 创建标签映射
    label_names = sorted(labels_df['label'].unique())
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    
    print(f"发现 {len(label_names)} 个类别: {label_names}")
    return label_to_idx

# 执行数据准备
label_to_idx = prepare_data()
class_names = list(label_to_idx.keys())
print("标签映射创建完成")


class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.image_files = []
        self.labels = []
        
        if mode == 'train':
            # 加载训练标签
            labels_path = os.path.join(KAGGLE_DATA_DIR, "trainLabels.csv")
            labels_df = pd.read_csv(labels_path)
            
            # 创建ID到标签的映射
            id_to_label = dict(zip(labels_df['id'], labels_df['label']))
            
            for img_file in os.listdir(root_dir):
                if img_file.endswith('.png'):
                    img_id = int(img_file.split('.')[0])
                    label = id_to_label.get(img_id)
                    if label is not None:
                        img_path = os.path.join(root_dir, img_file)
                        self.image_files.append(img_path)
                        self.labels.append(label_to_idx[label])
            print(f"训练集加载完成: {len(self.image_files)} 张图片")
        else:  # test模式
            for img_file in os.listdir(root_dir):
                if img_file.endswith('.png'):
                    img_path = os.path.join(root_dir, img_file)
                    self.image_files.append(img_path)
                    self.labels.append(-1)  # 测试集无标签
            print(f"测试集加载完成: {len(self.image_files)} 张图片")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        # 如果是测试集，返回图像和文件名
        if self.mode == 'test':
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            return image, img_id
        else:
            return image, label

print("自定义数据集类定义完成")


def get_augmentation():
    # 训练集增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 验证集和测试集转换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    return train_transform, test_transform

def create_data_loaders(batch_size=128):
    train_transform, test_transform = get_augmentation()
    
    # 创建数据集
    print("创建训练/验证数据集...")
    train_val_dataset = CIFAR10Dataset(
        root_dir=train_dir,
        transform=train_transform,
        mode='train'
    )
    
    # 分割训练集和验证集 (80%训练, 20%验证)
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_set, val_set = random_split(
        train_val_dataset, [train_size, val_size]
    )
    print(f"训练集大小: {len(train_set)}, 验证集大小: {len(val_set)}")
    
    # 测试集
    print("创建测试数据集...")
    test_set = CIFAR10Dataset(
        root_dir=test_dir,
        transform=test_transform,
        mode='test'
    )
    print(f"测试集大小: {len(test_set)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader, test_loader

# 创建数据加载器
batch_size = 128
train_loader, val_loader, test_loader = create_data_loaders(batch_size)
print("数据加载器创建完成")


class CIFAR10Model(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 初始化模型
model = CIFAR10Model(num_classes=len(class_names))
print("模型初始化完成")
print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
    """训练模型并记录性能指标"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = model.to(device)
    
    # 记录训练过程
    train_losses = []
    train_accuracies = []  # 训练准确率
    val_losses = []
    val_accuracies = []  # 验证准确率
    
    best_val_acc = 0.0
    best_model = None
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0  # 训练正确样本数
        total_samples = 0    # 总样本数
        
        # 训练阶段
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        # 计算训练指标
        epoch_loss = running_loss / total_samples
        epoch_acc = 100.0 * running_correct / total_samples
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 计算验证指标
        val_loss = val_loss / val_total
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 计算时间
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # 打印进度
        print(f'Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s')
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%')
        
        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model = model.state_dict()
            best_model_path = os.path.join(WORKING_DIR, 'best_cifar10_model.pth')
            torch.save(best_model, best_model_path)
            print(f"保存最佳模型到 {best_model_path}, 验证准确率: {val_accuracy:.2f}%")
    
    # 最终保存模型
    final_model_path = os.path.join(WORKING_DIR, 'final_cifar10_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型保存到 {final_model_path}")
    
    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
        print("加载最佳模型用于测试")
    
    return model, train_losses, train_accuracies, val_losses, val_accuracies


def plot_training_curves(train_losses, train_accuracies, val_accuracies, num_epochs):
    """绘制训练曲线图"""
    plt.figure(figsize=(12, 6))
    
    # 1. 训练损失图表
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks([5, 10, 15, 20] if num_epochs >= 20 else 
               [5, 10, 15, 20][:num_epochs//5])  # 设置X轴刻度
    
    # 设置Y轴范围和刻度
    max_loss = max(train_losses)
    plt.ylim(0, max_loss * 1.1)
    plt.yticks(np.linspace(0, max_loss, 5))
    
    # 2. 准确率图表
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, 'g-s', linewidth=2, markersize=8, label='Train Acc')
    plt.plot(range(1, num_epochs+1), val_accuracies, 'r-^', linewidth=2, markersize=8, label='Valid Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.xticks([5, 10, 15, 20] if num_epochs >= 20 else 
               [5, 10, 15, 20][:num_epochs//5])  # 设置X轴刻度
    
    # 设置Y轴范围和刻度
    min_acc = min(min(train_accuracies), min(val_accuracies))
    plt.ylim(max(0, min_acc - 5), 100)
    plt.yticks(range(0, 101, 20))
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(WORKING_DIR, 'training_curves.png'))
    plt.show()

print("训练函数定义完成")


# 超参数设置
num_epochs = 20
learning_rate = 0.001

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

print("开始训练模型...")
print(f"训练周期: {num_epochs}")
print(f"学习率: {learning_rate}")
print(f"批量大小: {batch_size}")

# 训练模型
model, train_losses, train_accuracies, val_losses, val_accuracies = train_model(
    model, train_loader, val_loader, 
    criterion, optimizer, scheduler, num_epochs
)

print("训练完成")


# 绘制训练曲线 
print("绘制训练曲线...")
plot_training_curves(train_losses, train_accuracies, val_accuracies, num_epochs)
print("训练曲线已保存")

# 生成测试集预测 
print("生成测试集预测...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

predictions = []
img_ids = []

with torch.no_grad():
    for i, (images, ids) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # 转换预测标签为类别名称
        for idx, pred in enumerate(predicted.cpu().numpy()):
            img_ids.append(ids[idx])
            predictions.append(class_names[pred])
        
        if (i + 1) % 10 == 0:
            print(f"已处理 {len(img_ids)}/{len(test_loader.dataset)} 张测试图片")

# 创建提交文件
submission_df = pd.DataFrame({
    'id': img_ids,
    'label': predictions
})

# 按id排序
submission_df['id'] = submission_df['id'].astype(int)
submission_df = submission_df.sort_values('id')

# 保存为CSV文件
submission_path = os.path.join(WORKING_DIR, 'submission.csv')
submission_df.to_csv(submission_path, index=False)
print(f"提交文件已保存为 '{submission_path}'")
print(f"前5行提交内容:\n{submission_df.head()}")


