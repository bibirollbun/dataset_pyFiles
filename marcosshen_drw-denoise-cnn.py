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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class Config:
    TRAIN_PATH = "/kaggle/input/denoise-drw/denoise_train_df.csv"
    TEST_PATH = "/kaggle/input/denoise-drw/denoise_test_df.csv"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    FEATURES = ['X21', 'X20', 'X28', 'X863', 'X29', 'X19', 'X27', 'X22', 'X858',
       'X219', 'X860', 'X531', 'X287', 'X289', 'X291', 'X293', 'X857',
       'X295', 'X598', 'X218', "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    """Load and preprocess data"""
    train_df = pd.read_csv(Config.TRAIN_PATH
                              )
    test_df = pd.read_csv(Config.TEST_PATH
                             )
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df


train_df,test_df,submission_df = load_data()


train_df.head()


test_df.head()


submission_df.head()


# train_df = train_df.replace([np.inf, -np.inf], np.nan)
# train_df = train_df.fillna(0)


# 1. 读入数据
df = train_df  # 假设已加载为 pandas.DataFrame
X = df.iloc[:, :-1].values  # (525887, 25)
y = df.iloc[:, -1].values  # (525887,)

# 2. 划分数据：先划分 train + temp，再从 temp 中再分出 val/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=0.90, random_state=42)
# temp 占 10%，我们再一分为二：各占 5%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42)

print("Split shapes:", X_train.shape, X_val.shape, X_test.shape)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
test_df = scaler.transform(test_df.values)


class PureConv1DRegressor(nn.Module):
    """纯一维卷积回归模型"""
    def __init__(self, input_dim=25, 
                 conv_channels=[64, 128, 256, 128],
                 kernel_sizes=[3, 5, 3, 3],
                 dilation_rates=[1, 2, 3, 1],
                 fc_dims=[128, 64],
                 dropout=0.2):
        """
        参数:
        input_dim: 输入特征维度 (默认25)
        conv_channels: 各卷积层的通道数 (默认[64, 128, 256, 128])
        kernel_sizes: 各卷积层的核大小 (默认[3, 5, 3, 3])
        dilation_rates: 各卷积层的膨胀率 (默认[1, 2, 3, 1])
        fc_dims: 全连接层维度 (默认[128, 64])
        dropout: Dropout率 (默认0.2)
        """
        super().__init__()
        
        # 输入层: 添加通道维度 [batch, 1, input_dim]
        self.input_layer = nn.Conv1d(
            in_channels=1, 
            out_channels=conv_channels[0],
            kernel_size=1
        )
        
        # 一维卷积块
        self.conv_blocks = nn.ModuleList()
        for i in range(len(conv_channels) - 1):
            conv_layer = nn.Conv1d(
                in_channels=conv_channels[i],
                out_channels=conv_channels[i+1],
                kernel_size=kernel_sizes[i],
                dilation=dilation_rates[i],
                padding=self._calculate_padding(kernel_sizes[i], dilation_rates[i])
            )
            
            block = nn.Sequential(
                conv_layer,
                nn.BatchNorm1d(conv_channels[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_blocks.append(block)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接预测头
        self.fc_layers = nn.ModuleList()
        fc_input_dim = conv_channels[-1]
        for dim in fc_dims:
            self.fc_layers.append(nn.Linear(fc_input_dim, dim))
            self.fc_layers.append(nn.BatchNorm1d(dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            fc_input_dim = dim
        
        # 最终输出层
        self.output = nn.Linear(fc_input_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def _calculate_padding(self, kernel_size, dilation):
        """计算保持特征长度不变的填充大小"""
        padding = (dilation * (kernel_size - 1)) // 2
        return padding
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        
        # 添加通道维度: [batch, 1, input_dim]
        x = x.unsqueeze(1)
        
        # 输入层处理
        x = self.input_layer(x)
        
        # 通过卷积块
        for block in self.conv_blocks:
            x = block(x)
        
        # 全局平均池化: [batch, channels, 1]
        x = self.global_pool(x)
        
        # 移除维度: [batch, channels]
        x = x.squeeze(-1)
        
        # 通过全连接层
        for layer in self.fc_layers:
            x = layer(x)
        
        # 输出预测
        return self.output(x)



class ResidualConv1DRegressor(nn.Module):
    """带残差连接的一维卷积回归模型"""
    def __init__(self, input_dim=25, 
                 base_channels=64,
                 num_blocks=4,
                 block_channels=[64, 128, 256, 128],
                 kernel_sizes=[3, 3, 3, 3],
                 dilation_rates=[1, 2, 3, 1],
                 fc_dims=[128, 64],
                 dropout=0.2):
        super().__init__()
        
        # 输入层
        self.input_conv = nn.Conv1d(
            in_channels=1, 
            out_channels=base_channels,
            kernel_size=1
        )
        
        # 残差卷积块
        self.res_blocks = nn.ModuleList()
        in_channels = base_channels
        
        for i in range(num_blocks):
            out_channels = block_channels[i]
            res_block = ResidualConvBlock(
                in_channels, 
                out_channels,
                kernel_size=kernel_sizes[i],
                dilation=dilation_rates[i],
                dropout=dropout
            )
            self.res_blocks.append(res_block)
            in_channels = out_channels
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接预测头
        self.fc_head = nn.Sequential(
            nn.Linear(in_channels, fc_dims[0]),
            nn.BatchNorm1d(fc_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.BatchNorm1d(fc_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fc_dims[1], 1)
        )
    
    def forward(self, x):
        # 添加通道维度: [batch, 1, input_dim]
        x = x.unsqueeze(1)
        
        # 输入卷积
        x = self.input_conv(x)
        
        # 通过残差块
        for block in self.res_blocks:
            x = block(x)
        
        # 全局池化
        x = self.global_pool(x)
        
        # 移除维度: [batch, channels]
        x = x.squeeze(-1)
        
        # 预测头
        return self.fc_head(x)


class ResidualConvBlock(nn.Module):
    """残差卷积块"""
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        
        # 主路径
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            padding=(dilation * (kernel_size - 1)) // 2,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, 
            kernel_size=kernel_size,
            padding=(dilation * (kernel_size - 1)) // 2,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        
        # 激活和正则化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        # 第一层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 第二层
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        out += residual
        out = self.relu(out)
        
        return out


# 修改后的训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=5):
    best_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # 早停机制
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve_epochs = 0
            best_model_state = model.state_dict()  # 保存最优模型状态
            torch.save(best_model_state, 'best_model.pth')  # 保存到文件
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f'Early stopping at epoch {epoch+1} with best val loss: {best_loss:.4f}')
                model.load_state_dict(best_model_state)  # 恢复最优模型
                break
    
    # 如果全程没有触发早停，确保返回的是最优模型
    if no_improve_epochs < patience:
        model.load_state_dict(best_model_state)
    
    return model


train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train).float().to(Config.device), 
        torch.tensor(y_train).float().unsqueeze(1).to(Config.device)
    ), 
    batch_size=1024, 
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_val).float().to(Config.device), 
        torch.tensor(y_val).float().unsqueeze(1).to(Config.device)
    ),
    batch_size=1024, 
    shuffle=False
)


loss_fn = nn.MSELoss()
model = ResidualConv1DRegressor(
        input_dim=25,
        base_channels=64,
        num_blocks=4,
        block_channels=[64, 128, 256, 128],
        kernel_sizes=[3, 3, 3, 3],
        dilation_rates=[1, 2, 3, 1],
        fc_dims=[128, 64],
        dropout=0.2
    ).to(Config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


trained_model = train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=200, patience=10)


y_pre_test = trained_model(torch.tensor(X_test).float().to(Config.device)).cpu().detach().numpy()
test_mse = mean_squared_error(y_test, y_pre_test)
test_rmse = np.sqrt(test_mse)
test_mape = mean_absolute_percentage_error(y_test, y_pre_test)
test_r2 = r2_score(y_test, y_pre_test)
print(f"Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}, Test MAPE: {test_mape:.4f}, Test R^2: {test_r2:.4f}")


torch.save(trained_model, 'denoise_CNN.pth')


# 将测试数据转换为PyTorch张量（仍在CPU上）
pre_tensor = torch.tensor(test_df).float()

# 创建Dataset和DataLoader进行分批
pre_dataset = TensorDataset(pre_tensor)
pre_loader = DataLoader(pre_dataset, batch_size=512, shuffle=False)  # 根据显存调整batch_size

predictions = []
trained_model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 禁用梯度计算节省显存
    for batch in pre_loader:
        inputs = batch[0].to(Config.device)  # 仅将当前批次送入GPU
        
        # 执行预测
        batch_pred = trained_model(inputs)
        
        # 立即移回CPU并释放GPU显存
        batch_pred = batch_pred.cpu().numpy()
        predictions.append(batch_pred)

        # 显式释放不再需要的GPU张量
        del inputs, batch_pred
        torch.cuda.empty_cache()  # 清空CUDA缓存

# 合并所有批次结果
y_pre = np.vstack(predictions)

# 保存结果
submission_df["prediction"] = y_pre
submission_df.to_csv("/kaggle/working/submission_denoise_CNN.csv",index=False)

