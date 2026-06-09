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


!pip install py7zr  # 安装解压 7z 依赖
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import collections
import math
import matplotlib.pyplot as plt  # 可视化依赖
from tqdm import tqdm  # 进度条依赖


class Accumulator:
    """累加器：统计损失、准确率等指标"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    """计算预测准确率（分类任务）"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)  # 取概率最大的类别
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """GPU 版准确率评估（支持多 GPU）"""
    if device is None:
        device = next(iter(net.parameters())).device
    net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Animator:
    """动态绘图类（训练过程可视化）"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), figsize=(8, 5)):
        plt.figure(figsize=figsize)
        self.axes = plt.gca()
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.fmts = fmts
        self.lines = []
        self.annotations = []
        self.curr_x = 0

    def add(self, x, y):
        """动态添加数据并更新图表"""
        if not hasattr(y, "__len__"):
            y = [y]
        if not self.lines:
            self.axes.set_xlabel(self.xlabel)
            self.axes.set_ylabel(self.ylabel)
            self.axes.set_xscale(self.xscale)
            self.axes.set_yscale(self.yscale)
            if self.xlim:
                self.axes.set_xlim(self.xlim)
            if self.ylim:
                self.axes.set_ylim(self.ylim)
            self.axes.legend(self.legend)
        
        for i, (yi, fmt) in enumerate(zip(y, self.fmts)):
            if i >= len(self.lines):
                line, = self.axes.plot([], [], fmt)
                self.lines.append(line)
            self.lines[i].set_data(list(self.lines[i].get_xdata()) + [x],
                                   list(self.lines[i].get_ydata()) + [yi])
        self.axes.relim()
        self.axes.autoscale_view()
        plt.draw()
        plt.pause(0.001)  # 短暂暂停以更新图表


import py7zr
# 数据集路径配置（Kaggle 输入路径 + 解压路径）
data_root = '/kaggle/input/cifar-10'  # Kaggle 竞赛数据集输入路径
unzip_dir = '/kaggle/working/cifar-10-unzipped'  # 解压到可写目录（Kaggle 工作区）

# 解压 train.7z
with py7zr.SevenZipFile(os.path.join(data_root, 'train.7z'), 'r') as z:
    z.extractall(path=unzip_dir)

# 解压 test.7z
with py7zr.SevenZipFile(os.path.join(data_root, 'test.7z'), 'r') as z:
    z.extractall(path=unzip_dir)

# 定义关键路径（解压后的数据目录）
train_dir = os.path.join(unzip_dir, 'train')  # 训练集原始图像目录
test_dir = os.path.join(unzip_dir, 'test')    # 测试集原始图像目录
train_labels_path = os.path.join(data_root, 'trainLabels.csv')  # 标签文件路径


def read_csv_labels(fname):
    """读取标签文件，返回 {文件名: 标签} 字典"""
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]  # 跳过表头
    tokens = [l.rstrip().split(',') for l in lines]
    return dict((name, label) for name, label in tokens)

def copyfile(filename, target_dir):
    """复制文件到目标目录（自动创建目录）"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    """拆分训练集为：训练集 + 验证集"""
    label_count = collections.Counter(labels.values())
    min_label_count = min(label_count.values())  # 样本最少的类别数量
    n_valid_per_label = max(1, math.floor(min_label_count * valid_ratio))  # 每个类别保留的验证集样本数

    label_count = {}
    train_valid_test_root = os.path.join(unzip_dir, 'train_valid_test')  # 新数据目录
    for train_file in tqdm(os.listdir(data_dir), desc="整理训练集"):
        label = labels[train_file.split('.')[0]]  # 假设文件名格式：xxxx.png
        fname = os.path.join(data_dir, train_file)
        
        # 复制到 train_valid（合并训练+验证集，用于最终训练）
        copyfile(fname, os.path.join(train_valid_test_root, 'train_valid', label))
        
        # 分配验证集样本
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(train_valid_test_root, 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(train_valid_test_root, 'train', label))
    return n_valid_per_label

def reorg_test(data_dir):
    """整理测试集目录结构"""
    train_valid_test_root = os.path.join(unzip_dir, 'train_valid_test')
    for test_file in tqdm(os.listdir(data_dir), desc="整理测试集"):
        copyfile(
            os.path.join(data_dir, test_file),
            os.path.join(train_valid_test_root, 'test', 'unknown')
        )

def reorg_cifar10_data(valid_ratio=0.1):
    """主函数：整理 CIFAR-10 数据集"""
    labels = read_csv_labels(train_labels_path)
    reorg_train_valid(train_dir, labels, valid_ratio)
    reorg_test(test_dir)

# 执行数据整理（仅需运行一次）
reorg_cifar10_data()


# 数据增强与归一化（训练集）
transform_train = transforms.Compose([
    transforms.Resize(40),  # 放大到 40x40
    transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),  # 随机裁剪 + 缩放
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize([0.4914, 0.4822, 0.4465],  # 归一化（CIFAR-10 均值/方差）
                         [0.2023, 0.1994, 0.2010])
])

# 测试集仅归一化（无增强）
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])
])

# 数据集加载（基于整理后的目录）
train_valid_test_root = os.path.join(unzip_dir, 'train_valid_test')
train_ds = datasets.ImageFolder(
    os.path.join(train_valid_test_root, 'train'),
    transform=transform_train
)
train_valid_ds = datasets.ImageFolder(
    os.path.join(train_valid_test_root, 'train_valid'),
    transform=transform_train
)
valid_ds = datasets.ImageFolder(
    os.path.join(train_valid_test_root, 'valid'),
    transform=transform_test
)
test_ds = datasets.ImageFolder(
    os.path.join(train_valid_test_root, 'test'),
    transform=transform_test
)

# 数据加载器（Batch + Shuffle）
batch_size = 32  # 根据 GPU 显存调整（Kaggle 免费 GPU 建议 32/64）
train_iter = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
train_valid_iter = DataLoader(train_valid_ds, batch_size, shuffle=True, drop_last=True)
valid_iter = DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
test_iter = DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)


def get_net():
    """构建模型：MobileNetV2（预训练 + 微调分类头）"""
    # 加载预训练 MobileNetV2
    net = torchvision.models.mobilenet_v2(pretrained=True)
    # 替换最后一层分类器（适配 CIFAR-10 的 10 分类）
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, 10)
    return net


def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    """训练函数：支持动态可视化 + 最终结果绘图"""
    # 优化器 + 学习率调度
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    
    num_batches = len(train_iter)
    # 动态绘图初始化
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'valid acc'],
                        figsize=(8, 5))
    
    # 多 GPU 支持
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    # 记录指标（用于最终绘图）
    train_loss_list = []
    train_acc_list = []
    valid_acc_list = []

    for epoch in range(num_epochs):
        net.train()  # 训练模式（启用 Dropout、BatchNorm 等）
        metric = Accumulator(3)  # 统计：损失总和、正确数、样本数

        # 进度条显示训练过程
        with tqdm(train_iter, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, (features, labels) in enumerate(pbar):
                features, labels = features.to(devices[0]), labels.to(devices[0])
                
                # 前向传播 + 反向传播
                trainer.zero_grad()
                y_hat = net(features)
                l = loss_fn(y_hat, labels).mean()
                l.backward()
                trainer.step()
                
                # 更新指标
                metric.add(l.item(), accuracy(y_hat, labels), labels.numel())
                
                # 进度条显示当前批次指标
                pbar.set_postfix({
                    'loss': f'{metric[0] / metric[2]:.3f}',
                    'acc': f'{metric[1] / metric[2]:.3f}'
                })

        # 计算当前 Epoch 指标
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # 验证集评估
        valid_acc = None
        if valid_iter is not None:
            valid_acc = evaluate_accuracy_gpu(net, valid_iter)
            valid_acc_list.append(valid_acc)
            animator.add(epoch + 1, (train_loss, train_acc, valid_acc))
        else:
            animator.add(epoch + 1, (train_loss, train_acc, None))

        # 更新学习率
        scheduler.step()

    # 训练结束后，绘制最终结果图（与示例图一致）
    plt.figure(figsize=(8, 5))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_loss_list, label='train loss', color='blue', linestyle='-')
    plt.plot(epochs, train_acc_list, label='train acc', color='pink', linestyle='--')
    if valid_iter is not None:
        plt.plot(epochs, valid_acc_list, label='valid acc', color='green', linestyle='-.')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.show()

    # 打印最终结果
    print(f'\n训练结束 | train loss: {train_loss:.3f}, train acc: {train_acc:.3f}')
    if valid_iter is not None:
        print(f'验证集 acc: {valid_acc:.3f}')

    return net


# 确保导入NumPy库
import numpy as np
import matplotlib.pyplot as plt
import torch

# 设备配置（自动检测 GPU/CPU）
devices = [torch.device('cuda')] if torch.cuda.is_available() else [torch.device('cpu')]

# ========== 新增：预测结果可视化函数 ==========
def visualize_predictions(model, dataset, class_names, num_samples=8):
    """可视化模型预测结果"""
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(16, 3))
    for i, idx in enumerate(indices):
        img, _ = dataset[idx]
        img_tensor = img.unsqueeze(0).to(devices[0])
        
        with torch.no_grad():
            pred = model(img_tensor)
            pred_class = pred.argmax(dim=1).item()
        
        # 反归一化图像用于显示
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])
        img_np = np.clip(img_np, 0, 1)
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"预测: {class_names[pred_class]}", fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# ========== 以下是你提供的代码 ==========
# 训练参数（可根据需求调整）
num_epochs, lr, wd = 20, 2e-4, 5e-4  # 轮次、学习率、权重衰减
lr_period, lr_decay = 4, 0.9         # 学习率衰减周期、衰减率

# ========== 第一阶段训练：训练集 + 验证集评估 ==========
print("开始第一阶段训练（训练集 + 验证集评估）...")
net = get_net()
net = train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 保存第一阶段模型
torch.save(net.state_dict(), 'cifar10_model_stage1.pth')

# ========== 第二阶段训练：合并训练+验证集（为测试集预测做准备） ==========
print("\n开始第二阶段训练（合并训练+验证集）...")
net_final = get_net()
net_final = train(net_final, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 保存最终模型
torch.save(net_final.state_dict(), 'cifar10_model_final.pth')

# ========== 测试集预测 + 生成提交文件 ==========
print("\n开始测试集预测...")
net_final.eval()  # 切换到评估模式
preds = []
with torch.no_grad(), tqdm(test_iter, desc="测试集预测") as pbar:
    for X, _ in pbar:
        y_hat = net_final(X.to(devices[0]))
        preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())

# 生成 Kaggle 提交文件
sample_submission = pd.read_csv(os.path.join(data_root, 'sampleSubmission.csv'))
sorted_ids = sample_submission['id'].tolist()  # 严格按示例提交文件的 ID 顺序

# 标签映射（修正后的完整代码）
label_to_idx = train_valid_ds.class_to_idx
idx_to_label = {v: k for k, v in label_to_idx.items()}  # 修复字典推导式

# 写入结果
with open('submission.csv', 'w') as f:
    f.write('id,label\n')
    for img_id, pred_idx in zip(sorted_ids, preds):
        f.write(f'{img_id},{idx_to_label[pred_idx]}\n')

print("提交文件已生成：submission.csv")

# ========== 可视化测试集预测结果 ==========
print("\n可视化测试集预测结果:")
visualize_predictions(net_final, test_ds, train_valid_ds.classes)

