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


import os
import zipfile
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==========================================
# 1. 环境配置与数据解压
# ==========================================

# 设置随机种子，保证结果可复现
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(42)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

# 定义路径
# Kaggle 输入数据通常在 /kaggle/input 下，是只读的
# 我们需要将其解压到 /kaggle/working/ 下
input_zip_path = '/kaggle/input/dogs-vs-cats/train.zip'
working_dir = '/kaggle/working/train_extracted'
train_dir = os.path.join(working_dir, 'train')

# 解压数据
if not os.path.exists(train_dir):
    print("正在解压数据，请稍候...")
    with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
        zip_ref.extractall(working_dir)
    print("数据解压完成！")
else:
    print("检测到数据已解压，跳过解压步骤。")

# 获取所有图片文件列表
all_files = os.listdir(train_dir)
image_files = [f for f in all_files if f.endswith('.jpg')]
print(f"图片总数: {len(image_files)}")

# ==========================================
# 2. 数据预处理与加载 (Dataset & DataLoader)
# ==========================================

# 定义图像转换：调整大小 -> 转张量 -> 归一化
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 统一调整为 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定义数据集类
class CatDogDataset(Dataset):
    def __init__(self, file_list, root_dir, transform=None):
        self.file_list = file_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # 打开图片
        image = Image.open(img_path)
        
        # 根据文件名解析标签: dog.x.jpg -> 1, cat.x.jpg -> 0
        if 'dog' in img_name:
            label = 1
        else:
            label = 0
        
        # 应用预处理
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 划分训练集 (80%) 和验证集 (20%)
random.shuffle(image_files)
split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# 创建数据集实例
train_dataset = CatDogDataset(train_files, train_dir, transform=transform)
val_dataset = CatDogDataset(val_files, train_dir, transform=transform)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集数量: {len(train_dataset)}, 验证集数量: {len(val_dataset)}")

# ==========================================
# 3. 建立分类模型 (CNN)
# ==========================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 128 -> 64
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64 -> 32
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 -> 16
        )
        
        # 分类器层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # 防止过拟合
            nn.Linear(512, 2) # 输出2类：猫, 狗
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN().to(device)
# print(model) # 如果需要查看模型结构可取消注释

# ==========================================
# 4. 模型训练与参数调优
# ==========================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化 history 字典，防止 NameError
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

def train_process(num_epochs=10):
    print(f"开始训练，共 {num_epochs} 轮...")
    
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()       # 清空梯度
            outputs = model(images)     # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward()             # 反向传播
            optimizer.step()            # 更新参数
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        # 记录数据
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")

# 执行训练
# Kaggle GPU 环境下建议设为 10-15
train_process(num_epochs=10)

# ==========================================
# 5. 结果分析与可视化
# ==========================================

# 绘制 Loss 和 Accuracy 曲线
def plot_results(hist):
    epochs_range = range(1, len(hist['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist['train_loss'], label='Train Loss')
    plt.plot(epochs_range, hist['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, hist['train_acc'], label='Train Acc')
    plt.plot(epochs_range, hist['val_acc'], label='Validation Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.show()

# 预测展示函数
def show_predictions(loader, count=5):
    model.eval()
    images_shown = 0
    
    plt.figure(figsize=(15, 4))
    
    # 获取一个 batch 的数据
    data_iter = iter(loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # 反归一化参数用于显示
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(3, 1, 1)
    
    for i in range(count):
        img = images[i]
        # 反归一化
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        # 转换为 numpy (H, W, C)
        img_np = img.cpu().permute(1, 2, 0).numpy()
        
        true_label = "Dog" if labels[i].item() == 1 else "Cat"
        pred_label = "Dog" if preds[i].item() == 1 else "Cat"
        color = 'green' if true_label == pred_label else 'red'
        
        plt.subplot(1, count, i+1)
        plt.imshow(img_np)
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
        
    plt.show()

print("\n正在绘制训练曲线...")
plot_results(history)

print("\n正在展示验证集预测结果...")
show_predictions(val_loader, count=6)

