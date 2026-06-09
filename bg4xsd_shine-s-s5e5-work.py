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


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义数据集类
class CaloriesDataset(Dataset):
    def __init__(self, features, targets):
        # 确保数据是float32类型
        self.features = torch.FloatTensor(features).float()
        self.targets = torch.FloatTensor(targets).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 定义神经网络模型
class CaloriesPredictor(nn.Module):
    def __init__(self, input_size):
        super(CaloriesPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LeakyReLU(0.1),  # 使用LeakyReLU替代ReLU
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(8, 1),
            nn.Softplus()  # 添加Softplus确保输出为正数
        )
        
        # 使用更保守的权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用较小的初始化范围
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # 添加输入检查
        if torch.isnan(x).any():
            print("警告: 输入包含NaN")
            x = torch.nan_to_num(x, nan=0.0)
        return self.model(x)
        
def prepare_data():
    # 读取数据
    train_data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
    
    # 数据清洗和预处理
    # 1. 首先删除id列
    train_data = train_data.drop(['id'], axis=1)
    
    # 2. 处理性别特征
    train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})
    
    # 3. 检查并处理异常值
    print("调整好的数据:")
    print(train_data.head())
    numeric_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
    
    # 4. 打印原始数据统计
    print("\n原始数据统计:")
    print(train_data[numeric_columns].describe())
    
    # 5. 使用更严格的异常值处理
    for column in numeric_columns:
        # 使用2.5倍标准差作为异常值界限
        mean = train_data[column].mean()
        std = train_data[column].std()
        lower_bound = mean - 2.5 * std
        upper_bound = mean + 2.5 * std
        
        # 打印异常值信息
        outliers = train_data[(train_data[column] < lower_bound) | (train_data[column] > upper_bound)]
        if len(outliers) > 0:
            print(f"\n{column} 的异常值数量: {len(outliers)}")
            print(f"异常值范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        train_data = train_data[(train_data[column] >= lower_bound) & (train_data[column] <= upper_bound)]
    
    # 6. 分离特征和目标
    X = train_data.drop(['Calories'], axis=1)
    y = train_data['Calories']
    
    # 7. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 检查标准化后的数据
    print("\n标准化后的数据统计:")
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    print(X_scaled_df.describe())
    
    # 8. 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据预处理完成:")
    print(f"- 训练集大小: {len(X_train)}")
    print(f"- 验证集大小: {len(X_val)}")
    print(f"- 特征数量: {X_train.shape[1]}")
    
    return X_train, X_val, y_train, y_val, scaler

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    learning_rates = []
    
    print(f"\n开始训练:")
    print(f"- 训练轮数: {num_epochs}")
    print(f"- 批次大小: {train_loader.batch_size}")
    print(f"- 初始学习率: {optimizer.param_groups[0]['lr']}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        batch_count = 0
        
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            # 检查输入数据
            if torch.isnan(features).any() or torch.isnan(targets).any():
                print(f"警告: 第 {epoch+1} 轮，批次 {batch_count} 的输入数据包含NaN")
                continue
            
            optimizer.zero_grad()
            outputs = model(features)
            
            # 检查输出
            if torch.isnan(outputs).any():
                print(f"警告: 第 {epoch+1} 轮，批次 {batch_count} 出现NaN输出")
                print("输入特征统计:", features.mean().item(), features.std().item())
                continue
            
            loss = criterion(outputs.squeeze(), targets)
            
            # 检查损失
            if torch.isnan(loss):
                print(f"警告: 第 {epoch+1} 轮，批次 {batch_count} 出现NaN损失")
                continue
            
            loss.backward()
            
            # 检查梯度
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"警告: 第 {epoch+1} 轮，批次 {batch_count}，参数 {name} 的梯度包含NaN")
                        param.grad.data.zero_()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                outputs = model(features)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 打印训练进度
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}')
            print(f'学习率: {current_lr:.6f}')
            print('-' * 50)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_model.pth')
            print(f'发现更好的模型！验证损失: {best_val_loss:.4f}')
    
    # 绘制训练过程
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制学习率曲线
    plt.subplot(1, 2, 2)
    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig('training_process.png')
    plt.close()
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print("训练过程图已保存到 'training_process.png'")


def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 准备数据
    X_train, X_val, y_train, y_val, scaler = prepare_data()
    
    # 创建数据加载器
    train_dataset = CaloriesDataset(X_train, y_train.values)
    val_dataset = CaloriesDataset(X_val, y_val.values)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 初始化模型并移动到GPU
    input_size = X_train.shape[1]
    model = CaloriesPredictor(input_size)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # 添加L2正则化
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )
    
    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    
    # 保存scaler供预测使用
    import joblib
    joblib.dump(scaler, 'scaler.pkl')


main()


import torch
import pandas as pd
import joblib


def load_model_and_scaler():
    # 1. 加载模型
    model = CaloriesPredictor(input_size=7)  # 7个特征
    # 2. 加载检查点（checkpoint）
    checkpoint = torch.load('best_model.pth')  # 加载整个检查点
    # 3. 提取模型的状态字典并加载
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)  # 将模型移动到GPU
    model.eval()
    
    # 加载scaler
    scaler = joblib.load('scaler.pkl')
    
    return model, scaler

def predict_calories(test_data_path, output_path):
    # 加载模型和scaler
    model, scaler = load_model_and_scaler()
    
    # 读取测试数据
    test_data = pd.read_csv(test_data_path)
    
    # 处理性别特征
    test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})
    
    # 准备特征
    X_test = test_data.drop(['id'], axis=1)
    X_test_scaled = scaler.transform(X_test)
    
    # 转换为tensor并移动到GPU
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    # 预测
    with torch.no_grad():
        predictions = model(X_test_tensor)
        # 将预测结果移回CPU
        predictions = predictions.cpu()
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'id': test_data['id'],
        'Calories': predictions.squeeze().numpy()
    })
    
    # 保存结果
    results.to_csv(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")


predict_calories('/kaggle/input/playground-series-s5e5/test.csv', 'submission.csv') 

