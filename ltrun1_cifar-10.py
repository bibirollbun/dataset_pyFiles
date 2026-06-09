import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path


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
    
    # CIFAR10的类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 随机选择样本
    indices = random.sample(range(len(x_data)), num_samples)
    
    # 创建图表
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    fig.suptitle('examples of train data', fontsize=14)
    
    for idx, ax in zip(indices, axes):
        # 获取图片和标签
        img = x_data[idx]
        img = img / 2 + 0.5
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
    "/kaggle/input/cnns-cifar-10/train_data.npz",  # 训练数据路径
    num_samples=4  # 显示4张图片
)



class CIFAR10Dataset(Dataset):
    """
    CIFAR10数据集加载器
    """
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __getitem__(self, idx):
        return {
            "images": torch.tensor(self._x[idx], dtype=torch.float32),
            "label": torch.tensor(self._y[idx], dtype=torch.long),
        }

    def __len__(self):
        return len(self._x)

def prepare_data_loader(
    path: str,
    ratio: float,
    batch_size: int,
    num_workers: int,
) -> dict:
    """
    参数:
        path (str): .npz格式的数据集文件路径
        ratio (float): 训练集比例
        batch_size (int): 批次大小
        num_workers (int): 数据加载的工作进程数
    返回:
        dict: 包含训练和测试数据加载器的字典
    """
    print("开始加载数据...")  # 添加调试信息
    train_data = np.load(path)
    print(f"数据文件已加载, 包含的keys: {train_data.files}")  # 显示数据内容
    
    x_data = train_data['x_train'].astype(np.float32) 
    y_data = train_data['y_train']
    print(f"数据形状 - x: {x_data.shape}, y: {y_data.shape}")  # 显示数据形状

    num_samples = len(x_data)
    split_idx = int(num_samples * ratio)
    x_train = x_data[:split_idx]
    y_train = y_data[:split_idx]
    x_test = x_data[split_idx:]
    y_test = y_data[split_idx:]
    print(f"训练集大小: {len(x_train)}, 测试集大小: {len(x_test)}")  # 显示划分结果

    train_batch_size = batch_size
    test_batch_size = batch_size
    
    print("创建数据加载器...")  # 添加调试信息
    train_dataset = CIFAR10Dataset(x_train, y_train)
    test_dataset = CIFAR10Dataset(x_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=1,  
        drop_last=True,
        pin_memory=True  
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=1,  
        pin_memory=True  
    )
    print("数据加载器创建完成")  # 添加调试信息

    return {"train": train_loader, "test": test_loader}



class SimpleCNN(nn.Module):
    """
    可配置的CNN模型
    
    参数:
        in_channels (int): 输入通道数
        num_classes (int): 分类类别数
        conv_layers (list): 每个卷积层的输出通道数列表
        fc_layers (list): 每个全连接层的输出维度列表
        kernel_size (int): 卷积核大小
        dropout_rate (float): Dropout比率
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        conv_layers: list = [32, 64],  # 默认两层卷积
        fc_layers: list = [128, 64],   # 默认两层全连接
        kernel_size: int = 3,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # 构建卷积层
        self.conv_blocks = nn.ModuleList()
        current_channels = in_channels
        
        for i, out_channels in enumerate(conv_layers):
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding='same'
                ),
                # nn.BatchNorm2d(out_channels), 可选择是否添加BatchNorm层
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_blocks.append(conv_block)
            current_channels = out_channels
           
            
        # 计算展平后的特征维度
        # 每经过一次MaxPool2d或AvgPool2d，特征图尺寸减半
        feature_size = current_channels * (32 // (2 ** len(conv_layers))) ** 2
        
        # 构建全连接层
        self.fc_blocks = nn.ModuleList()
        current_dim = feature_size
        
        for fc_dim in fc_layers:
            fc_block = nn.Sequential(
                nn.Linear(current_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.fc_blocks.append(fc_block)
            current_dim = fc_dim
            
        # 输出层
        self.output_layer = nn.Linear(current_dim, num_classes)

    def forward(self, x):
         # 通过所有卷积层
         for conv_block in self.conv_blocks:
             x = conv_block(x)
    
         # 展平
         x = torch.flatten(x, start_dim=1)
    
         # 通过所有全连接层
         for fc_block in self.fc_blocks:
              x = fc_block(x)
        
         # 输出层
         x = self.output_layer(x)
         return x 
        



class SimpleMLP(nn.Module):
    """
    可配置的多层感知机模型
    
    参数:
        input_size (int): 输入特征维度 (对于CIFAR10是3*32*32=3072)
        num_classes (int): 分类类别数
        hidden_layers (list): 每个隐藏层的输出维度列表
        dropout_rate (float): Dropout比率
        activation (str): 激活函数类型 ('relu', 'tanh', 'sigmoid')
    """
    def __init__(
        self,
        input_size: int = 3072,  # 3*32*32 for CIFAR10
        num_classes: int = 10,
        hidden_layers: list = [512, 256, 128],  # 默认三层隐藏层
        dropout_rate: float = 0.1,
        activation: str = 'relu'
    ):
        super(SimpleMLP, self).__init__()
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
            
        # 构建隐藏层
        self.fc_blocks = nn.ModuleList()
        current_dim = input_size
        
        for hidden_dim in hidden_layers:
            fc_block = nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                self.activation,
                nn.BatchNorm1d(hidden_dim),  # 添加批归一化
                nn.Dropout(dropout_rate)
            )
            self.fc_blocks.append(fc_block)
            current_dim = hidden_dim
            
        # 输出层
        self.output_layer = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        # 展平输入
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # 通过所有隐藏层
        for fc_block in self.fc_blocks:
            x = fc_block(x)
            
        # 输出层
        x = self.output_layer(x)
        return x
        


import torch.optim as optim

loss_fn = nn.CrossEntropyLoss()


def train_step(
    model: nn.Module, optimizer, batch: dict, device: torch.device
):
    """
    单步训练
    """
    batch_images = batch["images"].to(device)
    labels = batch["label"].to(device)
        
    optimizer.zero_grad()

    logits = model(batch_images) # 模型正向过程
        
    loss = loss_fn(logits, labels) # 计算总损失
        
    loss.backward() # 反向传播
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 添加梯度裁剪，防止梯度爆炸
        
    optimizer.step()

    return loss.item(), logits, labels
        

def eval_step(model: nn.Module, batch: dict, device: torch.device):
    # 单步评估
    model.eval()
    with torch.no_grad():
        batch_images = batch["images"].to(device)
        labels = batch["label"].to(device) 
        logits = model(batch_images) # 模型正向过程
        loss = loss_fn(logits, labels) # 计算总损失
        return loss.item(), logits, labels
    
def train_per_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch_size: int,
    train_loader: DataLoader,
    device: torch.device,
):
    model.train()
    num_data = len(train_loader.dataset)
    num_batches = len(train_loader)
    correct = 0
    total_loss = 0
    print(f"开始训练 - 总样本数: {num_data}")
    print(f"总批次数: {len(train_loader)}")
    
    for batch_idx, batch in enumerate(train_loader):
        loss, logits, labels = train_step(model, optimizer, batch, device)
        total_loss += loss
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()    
        if batch_idx % 50 == 0:  # 改为每50个批次打印一次
            current = batch_idx * batch_size + len(batch["images"])
            print(f"批次 {batch_idx}: Loss: {loss:>6.4f}, 进度: {current:>5d}/{num_data:>5d}")
    accuracy = correct / num_data
    avg_loss = total_loss / num_batches
    print(f"Train Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>.8f} \n")
        
def test_per_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
):
    """
    每轮测试
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    num_batches = len(test_loader)
    num_data = len(test_loader.dataset)
    with torch.no_grad():
        for batch in test_loader:
            loss, logits, labels = eval_step(model, batch, device)
            total_loss += loss
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / num_batches
    accuracy = correct / num_data
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>.8f} \n")
    return 100*accuracy





def controller(
    seed: int,
    model_type: str,  # 'cnn' 或 'mlp'
    # CNN特有参数
    in_channels: int = None,
    conv_layers: list = None,
    kernel_size: int = None,
    fc_layers  : list = None,
    # MLP特有参数
    input_size: int = None,
    hidden_layers: list = None,
    activation: str = None,
    # 通用参数
    num_classes: int = 10,
    dropout_rate: float = 0.1,
    data_path: Path = None,
    ratio: float = 0.8,
    batch_size: int = 64,
    num_workers: int = 4,
    epochs: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 0.004
):
    """
    训练控制器
    
    参数:
        model_type: 选择模型类型 ('cnn' 或 'mlp')
        其他参数见各模型的文档
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    
    if model_type == 'cnn':
        model = SimpleCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            conv_layers=conv_layers,
            fc_layers=fc_layers,  
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        ).to(device)
    elif model_type == 'mlp':
        model = SimpleMLP(
            input_size=input_size,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            activation=activation
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  #选择Adam优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs) # 选择余弦退火模型
    loader_dict = prepare_data_loader(path=data_path, ratio=ratio, batch_size=batch_size, num_workers=2 )
    train_loader = loader_dict["train"]
    test_loader = loader_dict["test"]
    print(f"Using device: {device}") 
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} \n--------------------------------")
        train_per_epoch(model, optimizer, batch_size, train_loader, device)
        acc = test_per_epoch(model, test_loader, device)
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")
    return model


# 随机数种子：确保每次运行代码得到相同的结果
seed = 42

# 模型类型：选择使用CNN（卷积神经网络）或MLP（多层感知机）
model_type = 'cnn'

# 卷积层配置：定义CNN模型中卷积层的结构
# [32, 64]表示有两层卷积层，第一层输出32个特征图，第二层输出64个特征图
# [32, 64, 128]则表示三层卷积层
conv_layers = [32, 64]

# 卷积核大小：定义每次卷积操作时，窗口的大小
# kernel_size=3表示使用3×3的滑动窗口
kernel_size = 3

# 全连接层配置：定义CNN末端全连接层的结构
# [128, 64]表示有两层全连接层，第一层有128个神经元，第二层有64个神经元
fc_layers = [128, 64]

# Dropout比率：随机"关闭"一部分神经元，防止模型过度依赖某些特征
# 比率0.2表示每次训练时随机关闭20%的神经元
dropout_rate = 0.2

#ratio比率：训练集中用于训练的数据量/总数据量
#比率0.8表示50000张图片中有40000张用于训练，10000用于测试
ratio = 0.8

# 训练轮数：整个训练数据集要被训练的次数
epochs = 50

# 批次大小：每次训练选取的图片数量
batch_size = 128

# 学习率：模型在训练过程中调整参数的步长
learning_rate = 0.001

#  CNN模型示例
model = controller(
    seed=seed,
    model_type=model_type,
    in_channels=3,
    conv_layers=conv_layers,  
    kernel_size=kernel_size,
    fc_layers=fc_layers,
    dropout_rate=dropout_rate,
    data_path=Path("/kaggle/input/cnns-cifar-10/train_data.npz"),
    ratio = ratio,
    epochs=epochs,  
    batch_size=batch_size,  
    learning_rate=learning_rate,
    weight_decay = 0.004 
)



# # 随机数种子：确保实验可以重复
# # 就像在相同的土地上，用相同的种子，收获相同的果实
# seed = 42

# # 模型类型：这里选择 MLP（多层感知机）模型
# model_type = 'mlp'

# # 输入维度：将32×32像素，3个颜色通道的图片展平后的大小
# # 计算方式：32 × 32 × 3 = 3072
# input_size = 3072

# # 隐藏层配置：定义神经网络中间层的结构
# # [1024, 512, 256] 表示有三层隐藏层，每层的神经元数量依次减少
# # 就像信息通过层层筛选，从复杂到简单的过程
# # - 第一层1024个神经元：接收大量原始特征
# # - 第二层512个神经元：进行中层特征提取
# # - 第三层256个神经元：提取更高层的抽象特征
# hidden_layers = [1024, 512, 256]

# # 激活函数：使用ReLU（修正线性单元）
# # ReLU的特点是：输入为正时保持不变，输入为负时输出为0
# activation = 'relu'

# # Dropout比率：随机"关闭"一部分神经元，防止模型过度依赖某些特征
# # 比率0.2表示每次训练时随机关闭20%的神经元
# dropout_rate = 0.2

# # 训练轮数：整个训练数据集要被训练的次数
# # 每个epoch会将所有训练数据都使用一遍
# epochs = 5

# # 批次大小：每次训练时同时处理的图片数量
# # 较大的batch_size训练更稳定，但需要更多内存
# # 较小的batch_size训练更快，但可能不够稳定
# batch_size = 32

# # 学习率：模型在训练过程中调整参数的步长
# # 较大的学习率学习速度快但可能错过最优解
# # 较小的学习率学习速度慢但更容易找到最优解
# learning_rate = 0.001
    
# #MLP模型示例
# model = controller(
#         seed=seed,
#         model_type='mlp',
#         input_size=input_size,  
#         hidden_layers=hidden_layers,
#         activation=activation,
#         dropout_rate=dropout_rate,
#         data_path=Path("/kaggle/input/cnns-cifar-10/train_data.npz"),
#         epochs=epochs,  
#         batch_size=batch_size,  
#         learning_rate=learning_rate
#     )



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
    
    # CIFAR10的类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
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
            img = x_data[idx]
            img = img/2 + 0.5
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
    "/kaggle/input/cnns-cifar-10/train_data.npz",  # 训练数据路径
    num_samples=4  # 显示4张图片
)


def evaluater(model: nn.Module, test_data_path: Path, solution_path: Path):
    """
    模型评估器
    """
    model.eval()
    with torch.no_grad():
        test_data = np.load(test_data_path)
        test_ids = test_data["ID"]
        test_images = test_data["x_test"].astype(np.float32) 
        test_images = torch.tensor(test_images)
        logits = model(test_images)
        _, predicted_labels = torch.max(logits, dim=1)
        predicted_labels = predicted_labels.cpu().numpy()
    predicted_labels = pd.DataFrame(predicted_labels, index=test_ids, columns=["label"])
    predicted_labels.to_csv(solution_path, index=True, index_label="ID")



evaluater(
    model,
    test_data_path=Path("/kaggle/input/cnns-cifar-10/test_data.npz"),
    solution_path=Path("/kaggle/working/submission.csv")
    )

