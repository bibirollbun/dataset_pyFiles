import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
class MNISTDataset(torch.utils.data.Dataset):
    """
    MNIST数据集加载器
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

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNIST是灰度图像，只有一个通道
])

def prepare_data_loader(
    ratio: float,
    train_batch_size: int,
    num_workers: int,
) -> dict:
    """
    参数:
        path (str): .npz格式的数据集文件路径
        ratio (float): 训练集比例
        train_batch_size (int): 批次大小
        num_workers (int): 数据加载的工作进程数
    返回:
        dict: 包含训练和测试数据加载器的字典
    """
    print("开始加载数据...")  # 添加调试信息
    train_dataset = datasets.MNIST(root='/kaggle/input/mnist-and-mlp/', train=True,  transform=transform)
    test_dataset = datasets.MNIST(root='/kaggle/input/mnist-and-mlp/', train=False,  transform=transform)
    train_loader = torch.utils.data.DataLoader(
                                                dataset=train_dataset, 
                                               batch_size=train_batch_size, 
                                               shuffle=True,
                                               drop_last=True,
                                               pin_memory=True  
                                              )
    test_loader = torch.utils.data.DataLoader(
                                              dataset=test_dataset, 
                                              batch_size=train_batch_size, 
                                              shuffle=False,
                                               drop_last=True,
                                               pin_memory=True  
    )


    print("数据加载器创建完成")  # 添加调试信息

    return {"train": train_loader, "test": test_loader}



class SimpleMLP(nn.Module):
    """
    可配置的多层感知机模型
    
    参数:
        input_size (int): 输入特征维度 (对于MNIST是1*28*28=784)
        num_classes (int): 分类类别数
        hidden_layers (list): 每个隐藏层的输出维度列表
        dropout_rate (float): Dropout比率
        activation (str): 激活函数类型 ('relu', 'tanh', 'sigmoid')
    """
    def __init__(
        self,
        input_size: int = 784,  
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
            print(f"Adding FC block with input dim {current_dim} and output dim {hidden_dim}")
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
from torch.utils.data import DataLoader

loss_fn = nn.CrossEntropyLoss()

def train_step(
    model: nn.Module, optimizer, batch: dict, device: torch.device
):
    """
    单步训练
    """
    batch_images, labels = batch
    batch_images = batch_images.to(device)
    labels = labels.to(device)
        
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
        batch_images, labels = batch
        batch_images = batch_images.to(device)
        labels = labels.to(device) 
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
            current = batch_idx * batch_size + len(batch[0])
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
    # MLP特有参数
    input_size: int = None,
    hidden_layers: list = None,
    activation: str = 'relu',
    
    # 通用参数
    num_classes: int = 10,
    dropout_rate: float = 0.1,
    ratio: float = 0.8,
    train_batch_size: int = 64,
    num_workers: int = 4,
    epochs: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 0.004
):
    """
    训练控制器
    
    参数:
        model_type: 选择模型类型 ('MLP' 或 'mlp')
        其他参数见各模型的文档
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    model = SimpleMLP(
            input_size=input_size,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            activation=activation
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  #选择Adam优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs) # 选择余弦退火模型
    loader_dict = prepare_data_loader(ratio=ratio, train_batch_size=train_batch_size, num_workers=2 )
    train_loader = loader_dict["train"]
    test_loader = loader_dict["test"]
    print(f"Using device: {device}") 
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} \n--------------------------------")
        train_per_epoch(model, optimizer, train_batch_size, train_loader, device)
        acc = test_per_epoch(model, test_loader, device)
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")
    return model


# 随机数种子：确保每次运行代码得到相同的结果
seed = 42

# 全连接层配置：定义MLP末端全连接层的结构
# [128, 64]表示有两层全连接层，第一层有128个神经元，第二层有64个神经元
hidden_layers = [512, 256, 128]

# Dropout比率：随机"关闭"一部分神经元，防止模型过度依赖某些特征
# 比率0.2表示每次训练时随机关闭20%的神经元
dropout_rate = 0.2

#ratio比率：训练集中用于训练的数据量/总数据量
#比率0.8表示50000张图片中有40000张用于训练，10000用于测试
ratio = 0.8

# 训练轮数：整个训练数据集要被训练的次数
epochs = 1

# 批次大小：每次训练选取的图片数量
batch_size = 128

# 学习率：模型在训练过程中调整参数的步长
learning_rate = 0.001

#  MLP模型示例
model = controller(
    seed=seed,
    input_size=1*28*28,  # MNIST数据集的输入大小
    hidden_layers = hidden_layers,
    dropout_rate=dropout_rate,
    ratio = ratio,
    epochs=epochs,  
    train_batch_size=batch_size,  
    learning_rate=learning_rate,
    weight_decay = 0.004 
)



import pandas as pd
from pathlib import Path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluater_with_dataloader_and_save(model, test_loader, device, solution_path=None):
    """
    使用 test_dataloader 计算准确率并可选择保存预测结果
    
    参数:
    - model: 需要评估的模型
    - test_loader: 测试数据加载器
    - device: 计算设备 (CPU 或 GPU)
    - solution_path: 保存预测结果的路径(可选)
    
    返回:
    - accuracy: 模型在测试集上的准确率
    """
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    all_ids = []
    all_predictions = []
    
    with torch.no_grad():  # 不计算梯度
        for images, labels in test_loader:
            # 如果 test_loader 包含ID信息，需要相应调整这里的解包方式
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 如果需要保存预测结果
            if solution_path is not None:
                # 假设我们能够获取ID信息
                batch_ids = range(len(all_predictions), len(all_predictions) + len(predicted))
                all_ids.extend(batch_ids)
                all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    print(f'测试集准确率: {accuracy:.4f}')
    
    # 保存预测结果
    if solution_path is not None:
        predictions_df = pd.DataFrame({"ID": all_ids, "label": all_predictions})
        predictions_df.to_csv(solution_path, index=False)
        print(f'预测结果已保存至 {solution_path}')
    
    return accuracy
loader_dict = prepare_data_loader(train_batch_size=128, num_workers=2,ratio=0.8 )
train_loader = loader_dict["train"]
test_loader = loader_dict["test"]
evaluater_with_dataloader_and_save(
    model,
    test_loader=test_loader,
    solution_path=Path("/kaggle/working/submission.csv"),
    device = device
    )




