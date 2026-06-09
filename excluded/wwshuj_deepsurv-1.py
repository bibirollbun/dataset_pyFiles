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



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import torch
import numpy as np
import random

# 设置随机种子
seed = 42

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)  # 如果使用多个GPU，设置所有GPU的种子

# 设置 PyTorch 数据加载器的随机种子
torch.backends.cudnn.deterministic = True  # 强制cuDNN使用确定性算法
torch.backends.cudnn.benchmark = False  # 禁用cuDNN的自动优化


# 读取数据
train_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

# train_data = train_data.drop(columns=['psych_disturb', 'arrhythmia', 'vent_hist', 'renal_issue', 'pulm_severe', 'rituximab', 'melphalan_dose', 'cardiac', 'pulm_moderate'])
# test_data = test_data.drop(columns=['psych_disturb', 'arrhythmia', 'vent_hist', 'renal_issue', 'pulm_severe', 'rituximab', 'melphalan_dose', 'cardiac', 'pulm_moderate'])

# 检查数值型列和类别型列
num_train_data = train_data.select_dtypes(include=[np.number])
cat_train_data = train_data.select_dtypes(exclude=[np.number])

num_test_data = test_data.select_dtypes(include=[np.number])
cat_test_data = test_data.select_dtypes(exclude=[np.number])

# 对数值型列填充均值
num_train_data = num_train_data.fillna( num_train_data.mean())
num_test_data = num_test_data.fillna( num_test_data.mean())

# 对类别型列填充众数
for col in cat_train_data.columns:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])

for col in cat_test_data.columns:
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])

# 将填充后的数据放回原数据框
train_data[num_train_data.columns] = num_train_data
test_data[num_test_data.columns] = num_test_data


# 特征与目标变量分离
X_train = train_data.drop(columns=['efs', 'efs_time', 'ID'])
y_train = train_data[['efs', 'efs_time']]
X_test = test_data.drop(columns=['ID'])


# 定义数值列和类别列
numeric_columns_X_train = X_train.select_dtypes(include=[np.number]).columns
non_numeric_columns_X_train = X_train.select_dtypes(include=[object]).columns

# 使用 ColumnTransformer 对类别变量进行独热编码
preprocessor_train = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_columns_X_train),
        ('cat', OneHotEncoder(), non_numeric_columns_X_train)  # 对非数值列进行独热编码
    ],
    remainder='drop'  # 丢弃未指定的列
)



# 构建一个包含预处理的管道
pipeline_train = Pipeline(steps=[
    ('preprocessor', preprocessor_train),
])

# 对训练数据进行预处理
X_train = pipeline_train.fit_transform(X_train)
X_test = pipeline_train.transform(X_test)


# X_train_dense = X_train.toarray()  # 将稀疏矩阵转换为密集矩阵
# X_test_dense = X_test.toarray()   # 将稀疏矩阵转换为密集矩阵

# 数据标准化
scaler = StandardScaler()  #标准化
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将 20% 的数据划分为验证集
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# 打印数据形状，确认划分是否正确
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")

# 将数据转为torch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)


print(f"X_train数据形状为: {(X_train_tensor).shape}")
print(f"y_train数据形状为: {(y_train_tensor).shape}")
print(f"X_val数据形状为：{(X_val_tensor).shape}")
print(f"y_val数据形状为：{(y_val_tensor).shape}")
print(f"X_test数据形状为: {(X_test_tensor).shape}")

print(y_train_tensor[:3])

# 创建自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, X_data, y_data):
        """
        初始化Dataset
        X_data: 特征数据（Tensor格式）
        y_data: 标签数据（Tensor格式）
        """
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        """返回数据集的大小"""
        return len(self.X_data)

    def __getitem__(self, idx):
        """根据索引返回单个样本"""
        x = self.X_data[idx]
        y = self.y_data[idx]
        return x, y

# 创建数据集
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
val_dataset = CustomDataset(X_val_tensor, y_val_tensor)
test_dataset = CustomDataset(X_test_tensor, None)  # 测试集没有标签

# # 设置batch_size
def batch_size(batch_size):
    batch_size = int(batch_size)
    return batch_size

batch_size = batch_size(32)


# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 查看数据加载器中的数据批次
for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
    print(f"Batch {batch_idx+1}")
    print(f"Feature batch shape: {batch_x.shape}")  # (batch_size, feature_size)
    print(f"Label batch shape: {batch_y.shape}")  # (batch_size, 2) -> 2是efs和efs_time
    break  # 打印第一个批次的数据形状并停止






import torch.nn as nn


class DeepSurv(nn.Module):
    ''' DeepSurv is a deep learning model for survival analysis. '''

    def __init__(self, input_dim, hidden_dims, activation, dropout):
        """
        Initialize the DeepSurv model.

        :param input_dim: (int) The number of input features.
        :param hidden_dims: (list of int) List of dimensions for hidden layers.
        :param activation: (str) The activation function to use ('ReLU', 'Tanh', etc.).
        :param dropout: (float) Dropout rate for regularization.
        """
        super(DeepSurv, self).__init__()

        self.activation = activation
        self.dropout = dropout

        # Build the neural network architecture
        self.layers = []

        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(self.get_activation(activation))
        self.layers.append(nn.Dropout(p=self.dropout))

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            self.layers.append(self.get_activation(activation))
            self.layers.append(nn.Dropout(p=self.dropout))

        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], 1))  # Only one output: the risk score

        self.model = nn.Sequential(*self.layers)

    def get_activation(self, name):
        """
        Return the activation function based on the provided name.
        """
        if name == 'ReLU':
            return nn.ReLU()
        elif name == 'Tanh':
            return nn.Tanh()
        elif name == 'Sigmoid':
            return nn.Sigmoid()
        elif name == 'LeakyReLU':
            return nn.LeakyReLU()
        elif name == 'SELU':
            return nn.SELU()
        else:
            raise ValueError("Unsupported activation function: {}".format(name))

    def forward(self, X):
        """
        Perform forward pass on the input data X.

        :param X: (torch.Tensor) The input feature tensor.
        :return: (torch.Tensor) The predicted risk scores.
        """
        return self.model(X)




import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor

# 定义训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

for (X_train_tensor, y_train_tensor) in train_loader:
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)

for (X_val_tensor, y_val_tensor) in val_loader:
    X_val_tensor = X_val_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)

# 设置训练配置
input_dim = 183  # 输入特征的维度
hidden_dims = [256, 128, 64, 32, 16, 4]  # 隐藏层的神经元数
activation = 'ReLU'  # 使用 激活函数
dropout = 0  # Dropout 率
epochs = 50  # 训练的轮数
# batch_size =   # 批次大小
learning_rate = 0.000008  # 学习率
weight_decay = 0.00001 # 通过weight_decay参数进行L2正则化


np.random.seed(42)
random.seed(42)
# 为 PyTorch 设置随机种子
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)  # 如果使用多个GPU，设置所有GPU的种子


# 创建模型
model = DeepSurv(input_dim=input_dim, hidden_dims=hidden_dims, activation=activation, dropout=dropout)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1) # 动态调整学习率
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 梯度裁剪

# C-index 计算函数

def concordance_index(y_true, y_pred):
    """
    计算C-index（Concordance Index），用于生存分析模型性能评估。

    :param y_true: (torch.Tensor) 真实生存时间和事件标志，形状为 (n_samples, 2)，
                   第一列是事件标志（1为事件发生，0为未发生），第二列是生存时间。
    :param y_pred: (torch.Tensor) 模型的预测风险评分，形状为 (n_samples, 1)。
    :return: C-index值，范围在 [0.5, 1] 之间。
    """
    event = y_true[:, 0]  # 事件标志
    survival_time = y_true[:, 1]  # 生存时间
    risk_scores = y_pred.squeeze()  # 风险评分

    # 获取事件发生的样本索引
    event_indices = torch.where(event == 1)[0]

    concordant_pairs = 0
    discordant_pairs = 0
    tied_pairs = 0

    # 遍历所有事件发生的样本对
    for i in event_indices:
        for j in range(len(survival_time)):
            if survival_time[j] > survival_time[i]:  # j 处于风险集合中
                if risk_scores[j] < risk_scores[i]:
                    concordant_pairs += 1
                elif risk_scores[j] > risk_scores[i]:
                    discordant_pairs += 1
                else:
                    tied_pairs += 1

    total_pairs = concordant_pairs + discordant_pairs + tied_pairs
    c_index = (concordant_pairs + 0.5 * tied_pairs) / total_pairs if total_pairs > 0 else 0.5

    return c_index
    
def stratified_cindex(y_true, y_pred, race_groups):
    # 按种族分组
    race_cindices = []
    for race in race_groups:
        # 提取当前种族的数据
        race_data = y_true[race_groups == race]
        race_pred = y_pred[race_groups == race]

        # 计算当前种族的C-index
        cindex = concordance_index(race_data, race_pred)
        race_cindices.append(cindex)

    # 计算分层C-index
    mean_cindex = np.mean(race_cindices)
    std_cindex = np.std(race_cindices)
    stratified_cindex = mean_cindex - std_cindex
    return stratified_cindex


# # Cox比例风险模型损失函数
class CoxLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super(CoxLoss, self).__init__()
        self.eps = eps

    def cox_ph_loss_sorted(self, y_pred, y_true, eps: float = 1e-7) -> Tensor:
        """Requires the input to be sorted by descending duration time.
        See DatasetDurationSorted.

        We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
        where h = exp(log_h) are the hazards and R is the risk set, and d is event.

        We just compute a cumulative sum, and not the true Risk sets. This is a
        limitation, but simple and fast.
        """
        # Prevent log(0) by adding epsilon to the prediction
        y_pred = torch.clamp(y_pred, min=self.eps)

        log_h = torch.log(y_pred)
        events = y_true[:, 0]
        if events.dtype is torch.bool:
            events = events.float()
        events = events.view(-1)
        log_h = log_h.view(-1)
        gamma = log_h.max()
        log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
        return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())

    def cox_ph_loss(self, y_pred, y_true, eps: float = 1e-7) -> Tensor:
        """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

        We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
        where h = exp(log_h) are the hazards and R is the risk set, and d is event.

        We just compute a cumulative sum, and not the true Risk sets. This is a
        limitation, but simple and fast.
        """
        log_h = torch.log(y_pred)
        durations = y_true[:, 1]
        events = y_true[:, 0]
        idx = durations.sort(descending=True)[1]
        events = events[idx]
        log_h = log_h[idx]
        return self.cox_ph_loss_sorted(y_pred, y_true, eps)

    def forward(self, y_pred, y_true) -> torch.Tensor:
        return self.cox_ph_loss(y_pred, y_true)




# 选择损失函数和优化器
cox_loss = CoxLoss()
cox_loss = cox_loss.to(device)


# # 添加tensorboard
# writer = SummaryWriter("./logs_train")

# 训练循环
total_train_steps = 0



# 初始化列表
Epoch = []
C_index = []
Stratified_C_index = []
Val_Loss = []

for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    
    print("-------第{}论训练开始--------".format(epoch + 1))
    running_loss = 0.0

    # 训练数据迭代
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()  # 清除之前的梯度

        # 前向传播
        outputs = model(batch_x)

        # 计算损失
        loss = cox_loss(outputs, batch_y)

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_train_steps += 1
        
        if (batch_idx + 1) % 100 == 0:  # 每100个批次输出一次训练信息

            print(
                f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / (batch_idx + 1):.4f}")
            
            # writer.add_scalar("loss", loss, total_train_steps)


    
    model.eval()  # 设置模型为评估模式
    total_test_steps = 0
    val_loss = 0.0
    with torch.no_grad():
        #计算val_loss
        for data in val_loader:
            features, preds = data
            features = features.to(device)
            preds = preds.to(device)
            outputs = model(features)
            loss = cox_loss(outputs, preds)
            val_loss += loss.item()
            total_test_steps += 1
        
        # 计算 C-index
        y_pred_train = model(X_val_tensor)  # 预测训练集的风险评分
        val_c_index = concordance_index(y_val_tensor, y_pred_train)

        #分层C-index
        # 提取验证集的索引
        val_indices = y_val.index
        # 使用这些索引从原始数据中提取race_group列
        race_group_val = train_data.loc[val_indices, 'race_group'].values
        race_group_val = race_group_val[:batch_size]
        # y_true和y_pred是模型的真实标签和预测结果，race_group是种族列
        stratified_cindex_score = stratified_cindex(y_val_tensor, y_pred_train, race_group_val)

        print(f"Epoch [{epoch + 1}/{epochs}] Val C-index: {val_c_index:.4f} ，Stratified C-index:{stratified_cindex_score}, Val_Loss：{val_loss / total_test_steps}")
        # writer.add_scalar("C-index", val_c_index, total_test_steps)
        # writer.add_scalar("Stratified C-index", stratified_cindex_score, total_test_steps)
        
        total_test_steps += 1
        # 动态更新 epoch 和 C-指数
        Epoch.append(epoch)
        C_index.append(val_c_index)
        Stratified_C_index.append(stratified_cindex_score)
        Val_Loss.append(val_loss / total_test_steps)


        torch.save(model, 'model_{}_gpu.pth'.format(epoch))
        print("模型已保存")

# 创建 DataFrame
C_index_df = pd.DataFrame({
    'Epoch': Epoch,
    'C-index': C_index,
    'Stratified_C_index': Stratified_C_index,
    'Val_Loss': Val_Loss
})

print(C_index_df)

# 保存为C-index.csv
C_index_df.to_csv('C_index.csv', index=False)



import os

import pandas as pd
import torch

# 定义训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
X_test_tensor = X_test_tensor.to(device)

model = torch.load("model_41_gpu.pth", weights_only=False)
# 切换到评估模式
model.eval()
model.to(device)

# 禁用梯度计算，进行预测
with torch.no_grad():
    risk_pred = model(X_test_tensor)  # 获取预测的风险得分

# 输出预测的风险得分
print("Predicted Risk Scores:", risk_pred)

risk_pred = risk_pred.cpu().numpy()


# 创建提交文件
submission = pd.DataFrame({
    'ID': test_data['ID'],
    'prediction': risk_pred.flatten()
})

# 保存为submission.csv
submission.to_csv('submission.csv', index=False)


