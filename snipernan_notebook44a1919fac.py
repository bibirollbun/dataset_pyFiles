# Modify this
COMPETITION_NAME = "ml-09290929"


import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from pathlib import Path
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Dataset


class TransformerBlock(nn.Module):
    """
    Transformer Block

    参数:
        embed_dim (int): 输入特征的维度
        num_heads (int): 注意力头的数量
        ff_dim (int): 前馈网络中的隐藏维度
        rate (float, optional): Dropout率(默认为0.1)

    输入:
        inputs (torch.Tensor): 形状为[batch_size, sequence_length, embed_dim]的输入张量

    输出:
        torch.Tensor: 与输入形状相同的转换后的特征
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        rate: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        self.dense_1 = nn.Linear(embed_dim, ff_dim)
        self.dense_2 = nn.Linear(ff_dim, embed_dim)

        self.layer_norm_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.dropout_1 = nn.Dropout(rate)
        self.dropout_2 = nn.Dropout(rate)

    def forward(self, inputs):
        attn_output, _ = self.attention(inputs, inputs, inputs)
        attn_output = self.dropout_1(attn_output)
        out1 = self.layer_norm_1(inputs + attn_output)

        ffn_output = self.dense_1(out1)
        ffn_output = F.relu(ffn_output)
        ffn_output = self.dense_2(ffn_output)
        ffn_output = self.dropout_2(ffn_output)

        return self.layer_norm_2(out1 + ffn_output)


class TokenAndPositionEmbedding(nn.Module):
    """
    
    参数:
        max_length (int): 支持的最大序列长度
        vocab_size (int): 词汇表大小
        embed_dim (int): 嵌入向量的维度

    输入:
        x (torch.Tensor): 形状为[batch_size, sequence_length]的整数数组(Token索引)

    输出:
        torch.Tensor: 形状为[batch_size, sequence_length, embed_dim]的嵌入向量
    """

    def __init__(self, max_length: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        pe = torch.zeros(max_length, embed_dim).cuda()
        positions = torch.arange(0, max_length, device=pe.device).unsqueeze(1)
        # div_term = (torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))).cuda()
        div_term = (1 / (10000**(torch.arange(0, embed_dim, 2).float() / embed_dim))).cuda()
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer("positional_encoding", pe)
        
        # self.pos_emb = nn.Embedding(max_length, embed_dim)

    def forward(self, x):
        maxlen = x.size(-1)
        # positions = self.pos_emb(positions)
        
        x = self.token_emb(x)
        pos_enc = self.positional_encoding[:maxlen,:].unsqueeze(0)
        return x + pos_enc


class Transformer(nn.Module):
    """
    参数:
        embed_dim (int): token嵌入的维度大小
        num_heads (int): 注意力头的数量
        ff_dim (int): 前馈网络中的隐藏层大小
        num_block(int): Transformer Block的数量
        maxlen (int): 最大输入序列长度
        vocab_size (int): 词汇表大小

    输入:
        x (torch.Tensor): 形状为[batch_size, sequence_length]的整数数组(Token索引)

    输出:
        torch.Tensor: 形状为[batch_size, 2]的浮点数组(两个类别的概率分布)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_blocks:int, 
        maxlen: int,
        vocab_size: int,
    ):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_blocks)
        ])
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(embed_dim, 20)
        self.dropout2 = nn.Dropout(0.1)
        self.dense2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.embedding_layer(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = torch.mean(x, dim=1)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = F.softmax(x, dim=-1)
        return x


class LSTM(nn.Module):
    def __init__(self, 
                vocab_size : int, 
                embedding_dim : int, 
                hidden_dim : int, 
                n_layers:int = 1,
                dropout:float = 0.2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            batch_first=True, 
                            dropout=dropout if n_layers > 1 else 0
                            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2) # 二分类问题
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden_t, _) = self.lstm(embedded)
        logits = self.fc(self.dropout(hidden_t[-1, :, :]))
        probs = F.softmax(logits, dim=-1)
        return probs



class IMDBDataset(Dataset):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __getitem__(self, idx):
        return {
            "encoded_indices": torch.tensor(self._x[idx], dtype=torch.long),
            "label": torch.tensor(self._y[idx], dtype=torch.long),
        }

    def __len__(self):
        return len(self._x)


def prepare_data_loader(
    path: str,
    ratio: float,
    batch_size: int,
    num_workers: int = 4,
) -> dict:
    """
    参数:
        path (str): .npz格式的数据集文件路径
        ratio (float, optional): 用于训练的数据比例(默认为0.8)
        seed (int, optional): 用于随机打乱的种子(默认为12)
        batch_size (int, optional): 训练批量大小(默认为128)
        num_workers (int, optional): 数据加载的工作进程数(默认为4)

    返回:
        dict: 包含训练和测试数据加载器的字典
    """
    train_data = np.load(path)

    x_data = train_data["x_train"]
    y_data = train_data["y_train"]

    num_samples = len(x_data)
    split_idx = int(num_samples * ratio)
    x_train = x_data[:split_idx]
    y_train = y_data[:split_idx]
    x_test = x_data[split_idx:]
    y_test = y_data[split_idx:]

    train_batch_size = batch_size
    test_batch_size = train_batch_size

    # 创建PyTorch数据集
    train_dataset = IMDBDataset(x_train, y_train)
    test_dataset = IMDBDataset(x_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return {"train": train_loader, "test": test_loader}


def train_step(
    model: nn.Module, 
    loss_fn: Callable, 
    optimizer: optim.Optimizer, 
    batch: dict, 
    device: torch.device
):
    """
    参数:
        model (nn.Module): 要训练的模型
        loss_fn (Callable): 损失函数
        optimizer (optim.Optimizer): 用于更新模型参数的优化器
        batch (dict): 包含训练数据的批次
        device (torch.device): 用于计算的设备(gpu or cpu)
    返回:
        float: 当前批次的损失值
    """
    model.train()
    optimizer.zero_grad()

    batch_tokens = batch["encoded_indices"].to(device)
    labels = batch["label"].to(device)

    loss = loss_fn(model(batch_tokens), labels)

    loss.backward()
    optimizer.step()

    return loss.item()


def eval_step(model: nn.Module, metric_fn: Callable, batch: dict, device: torch.device):
    """
    参数:
        model (nn.Module): 用于测试的模型
        metric_fn (Callable): 测试集上的度量函数
        batch (dict): 测试数据的批次
        device (torch.device): 用于计算的设备(gpu or cpu)

    返回:
        tuple: (loss, logits, labels) - 损失值、预测结果和真实标签
    """

    model.eval()
    with torch.no_grad():
        batch_tokens = batch["encoded_indices"].to(device)
        labels = batch["label"].to(device)
        logits = model(batch_tokens)
        metric = metric_fn(logits, labels)

        return metric.item(), logits, labels


def train_per_epoch(
    model: nn.Module,
    loss_fn: Callable,
    optimizer: optim.Optimizer,
    batch_size: int,
    train_loader: DataLoader,
    device: torch.device,
):
    """
    参数:
        model (nn.Module): 训练模型
        optimizer (optim.Optimizer): 用于更新模型参数的优化器
        train_loader (DataLoader): 包含训练数据的DataLoader
        device (torch.device): 用于计算的设备
    """
    model.train()
    length = len(train_loader.dataset)
    for batch_idx, batch in enumerate(train_loader):
        loss = train_step(model, loss_fn, optimizer, batch, device)
        if batch_idx % 20 == 0:
            current = batch_idx * batch_size + len(batch["encoded_indices"])
            print(f" Loss: {loss:>6.4f}, {current:>5d}/{length:>5d}")


def test_per_epoch(
    model: nn.Module,
    metric_fn: Callable,
    test_loader: DataLoader,
    device: torch.device,
):
    """
    参数:
        model (nn.Module): 测试模型
        test_loader (DataLoader): 包含测试数据的DataLoader
        device (torch.device): 用于计算的设备
    """
    model.eval()

    total_loss = 0.0
    correct = 0

    num_batches = len(test_loader)
    num_data = len(test_loader.dataset)
    with torch.no_grad():
        for batch in test_loader:
            loss, logits, labels = eval_step(model, metric_fn, batch, device)
            total_loss += loss
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / num_batches
    accuracy = correct / num_data
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")


def controller(seed: int,
               model_type: str,
               embed_dim: int, 
               num_heads: int, 
               ff_dim: int,
               num_blocks:int,
               lstm_embed: int,
               hidden_dim: int,
               ratio: float,
               batch_size: int, 
               epochs: int, 
               learning_rate: float):
    torch.manual_seed(seed)
    data_path = Path(f"/kaggle/input/{COMPETITION_NAME}/processed_imdb_train_data.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "Transformer":
        model = Transformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_blocks=num_blocks,
            maxlen=200,
            vocab_size=20000,
        ).to(device)
    elif model_type == "LSTM":
        model = LSTM(
            vocab_size=20000,
            embedding_dim=lstm_embed,
            hidden_dim=hidden_dim,
        ).to(device)
    else:
        raise ValueError("Invalid model type")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = nn.CrossEntropyLoss()

    loader_dict = prepare_data_loader(data_path, ratio, batch_size)
    train_loader = loader_dict["train"]
    test_loader = loader_dict["test"]

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} \n--------------------------------")
        train_per_epoch(model, loss_fn, optimizer, batch_size, train_loader, device)
        scheduler.step() # 每个epoch结束时更新学习率
        test_per_epoch(model, metric_fn, test_loader, device)
    print("Done!")
    return model



seed = 12 # 设置我们的全局随机种子

model_type = "LSTM" # 指定要选择的模型["Transformer", "LSTM"]
# Transformer模型相关的超参数, 如果是LSTM则不需要关注
embed_dim = 32
num_heads = 2
ff_dim = 32
num_blocks = 2

# LSTM模型相关的超参数
lstm_embed = 32
hidden_dim = 32

ratio = 0.8 #训练集占数据集的比例
batch_size = 64 # batch的大小
epochs = 5 # 训练回合数
learning_rate = 1e-3 #初始学习率的大小


model = controller(seed,
                   model_type,
                   embed_dim,
                   num_heads, 
                   ff_dim,
                   num_blocks,
                   lstm_embed,
                   hidden_dim,
                   ratio,
                   batch_size,
                   epochs, 
                   learning_rate)


def evaluater(model: nn.Module):
    model.eval()
    test_data_path = Path(f"/kaggle/input/{COMPETITION_NAME}/processed_imdb_test_data.npz")
    submission_path = Path("/kaggle/working/submission.csv")
    with torch.no_grad():
        test_data = np.load(test_data_path)
        test_ids = test_data["ID"]
        test_tokens = test_data["x_test"]
        for i in range(5):
            batch_test_tokens = torch.tensor(test_tokens[5000*i: 5000*(i+1), :], dtype=torch.long).cuda()
            probs = model(batch_test_tokens)
            _, batch_predicted_labels = torch.max(probs, dim=-1)
            if i == 0:
                predicted_labels = batch_predicted_labels.cpu().numpy()
            else:
                predicted_labels = np.hstack((predicted_labels, batch_predicted_labels.cpu().numpy()))

    predicted_labels = pd.DataFrame(predicted_labels, index=test_ids, columns=["label"])
    predicted_labels.to_csv(submission_path, index=True, index_label="ID")
evaluater(model)


