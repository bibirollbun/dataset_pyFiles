import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")





import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def visualize_train_images(train_data_path, num_samples=4):
    """
    可视化训练集图片
    
    参数:
        train_data_path: 训练数据路径
        num_samples: 要显示的样本数量
    """
    # 加载数据
    data = np.load(train_data_path)
    x_data = data['x_train']
    y_data = data['y_train']
    
    # CIFAR100的类别名称
    classes = ['apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 
'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 
'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn mower', 
'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak tree', 'orange', 'orchid', 
'otter', 'palm tree', 'pear', 'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 
'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 
'sunflower', 'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 
'whale', 'willow tree', 'wolf', 'woman', 'worm']
    
    # 随机选择样本
    indices = random.sample(range(len(x_data)), num_samples)
    
    # 创建图表
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    fig.suptitle('examples of train data', fontsize=14)
    
    for idx, ax in zip(indices, axes):
        # 获取图片和标签
        img = x_data[idx]/256
        true_label = y_data[idx]
        
        # 显示图片
        img_display = np.transpose(img, (1, 2, 0))  # 转换通道顺序为(H,W,C)
        ax.imshow(img_display)
        
        # 添加标题：真实标签
        ax.set_title(f'{classes[true_label]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用方法
visualize_train_images(
    "/kaggle/input/image-classification-20251121/train_data.npz",  # 训练数据路径
    num_samples=4  # 显示4张图片
)


def load_cifar100_data_cnn(train_file, test_file):
    """从npz文件中加载并预处理CIFAR-100数据"""
    # 加载训练数据
    train_data = np.load(train_file)
    X_train = train_data['x_train']
    y_train = train_data['y_train']
    
    # 加载测试数据
    test_data = np.load(test_file)
    X_test = test_data['x_test']
    test_ids = test_data['ID']
    
    # 将图像数据从uint8转换为float32并归一化到[0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    return X_train, y_train, X_test, test_ids

def load_cifar100_data_mlp(train_file, test_file):
    """从npz文件中加载并预处理CIFAR-100数据"""
    # 加载训练数据
    train_data = np.load(train_file)
    X_train = train_data['x_train']
    y_train = train_data['y_train']
    
    # 加载测试数据
    test_data = np.load(test_file)
    X_test = test_data['x_train']
    test_ids = np.array(range(len(X_test)))
    
    # 将图像数据从(样本数, 3, 32, 32)展平为(样本数, 3072)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, test_ids


class CNN(nn.Module):
    """卷积神经网络模型"""
    def __init__(self, num_classes=100):
        super(CNN, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # 第二个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # 第三个卷积块
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二个卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 第三个卷积块
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x

class MLP(nn.Module):
    """简单的多层感知机模型"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


def train_model(model, X_train, y_train, X_val, y_val, num_classes, num_epochs=50, batch_size=64, learning_rate=0.001):
    """训练CNN模型"""
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # 训练模型
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        train_losses.append(epoch_loss)
        val_accuracies.append(epoch_acc)
        
        # 更新学习率
        scheduler.step(epoch_acc)
        
        # 保存最佳模型
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.2f}%, Best: {best_val_acc:.2f}%')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_cnn_model.pth'))
    
    return model, train_losses, val_accuracies


def predict(model, X_test, batch_size=128):
    """使用训练好的模型进行预测"""
    model.eval()
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for inputs in tqdm(test_loader, desc='Predicting'):
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    
    return np.array(predictions)


# 加载数据（CNN）
print("加载CIFAR-100数据...")
X_train, y_train, X_test, test_ids = load_cifar100_data_cnn(
    '/kaggle/input/image-classification-20251121/train_data.npz',
    '/kaggle/input/image-classification-20251121/test_data.npz'
)

print(f"训练数据形状: {X_train.shape}, 训练标签形状: {y_train.shape}")
print(f"测试数据形状: {X_test.shape}")

# 将训练数据分为训练集和测试集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"训练集形状: {X_train.shape}, 测试集形状: {X_val.shape}")

# 初始化模型（CNN）
num_classes=100
model = CNN(num_classes).to(device)

# 训练模型
print("训练CNN模型...")
num_epochs=50
batch_size=64
learning_rate=0.001

model, train_losses, val_accuracies = train_model(model,
    X_train, y_train, X_val, y_val,
    num_classes,
    num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate
)


import matplotlib.pyplot as plt
import numpy as np
import random

def visualize_predictions(model, train_data_path, num_samples=4,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    可视化模型预测结果
    
    参数:
        model: 训练好的模型
        train_data_path: 训练数据路径
        num_samples: 要显示的样本数量
    """
    # 加载数据
    data = np.load(train_data_path)
    x_data = data['x_train']
    y_data = data['y_train']
    
    # CIFAR100的类别名称
    classes = ['apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 
'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 
'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn mower', 
'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak tree', 'orange', 'orchid', 
'otter', 'palm tree', 'pear', 'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 
'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 
'sunflower', 'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 
'whale', 'willow tree', 'wolf', 'woman', 'worm']
    
    # 随机选择样本
    indices = random.sample(range(len(x_data)), num_samples)
    
    # 创建图表
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    fig.suptitle('visualization results', fontsize=14)
    
    # 设置模型为评估模式
    model.eval()
    
    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            # 获取图片和标签
            img = x_data[idx]/256
            true_label = y_data[idx]
            
            # 预处理图片并进行预测
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            pred_label = predicted.item()
            
            # 显示图片
            img_display = np.transpose(img, (1, 2, 0))  # 转换通道顺序为(H,W,C)
            ax.imshow(img_display)
            
            # 添加标题：真实标签 vs 预测标签
            ax.set_title(f'true: {classes[true_label]}\n pred: {classes[pred_label]}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用方法
visualize_predictions(
    model,  # 你训练好的模型
    "/kaggle/input/image-classification-20251121/train_data.npz",  # 训练数据路径
    num_samples=4  # 显示4张图片
)


# 在测试数据上进行预测
print("在测试数据上进行预测...")
predictions = predict(model, X_test)

# 创建submission.csv文件
print("创建submission.csv文件...")
submission_df = pd.DataFrame({
    'ID': test_ids,
    'labels': predictions
})

submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("submission.csv文件已保存")




