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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import gc
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

warnings.filterwarnings('ignore')

# 检查 CUDA 可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 数据读取函数
def read_data():
    print('Reading files...')
    train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
    test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
    train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
    test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
    sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
    print('Data read')
    return train_transaction, test_transaction, train_identity, test_identity, sample_submission

# 合并数据函数 - 优化合并逻辑
def merge_data(train_transaction, test_transaction, train_identity, test_identity):
    print('Merging data...')
    
    # 规范化列名
    train_identity.columns = [col.replace('_', '-') for col in train_identity.columns]
    test_identity.columns = [col.replace('_', '-') for col in test_identity.columns]
    
    # 使用左连接合并数据
    train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
    
    print(f'Train shape: {train.shape}, Test shape: {test.shape}')
    
    # 释放内存
    del train_transaction, test_transaction, train_identity, test_identity
    gc.collect()
    
    return train, test

def check_column_consistency(train, test, target_column='isFraud'):
    print('Checking column consistency between train and test sets...')
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    train_only_cols = train_cols - test_cols
    test_only_cols = test_cols - train_cols
    
    if train_only_cols:
        print(f"警告: 训练集有 {len(train_only_cols)} 个列是测试集没有的:")
        print(sorted(train_only_cols))
    else:
        print("训练集没有额外的列")
        
    if test_only_cols:
        print(f"警告: 测试集有 {len(test_only_cols)} 个列是训练集没有的:")
        print(sorted(test_only_cols))
    else:
        print("测试集没有额外的列")
    
    common_cols = train_cols.intersection(test_cols)
    common_ratio = len(common_cols) / max(len(train_cols), len(test_cols))
    print(f"共同列比例: {common_ratio:.2%}")
    
    if target_column in train_only_cols:
        print(f"确认: 目标列 '{target_column}' 只存在于训练集中")
    
    print("列名检查完成")
    return train_only_cols, test_only_cols

# 优化的日期特征处理
def process_dates(train, test):
    print('Processing dates...')
    
    # 从TransactionDT提取有意义的时间特征
    # 参考: https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203
    train['TransactionDT_day'] = train['TransactionDT'] // (24 * 60 * 60)
    test['TransactionDT_day'] = test['TransactionDT'] // (24 * 60 * 60)
    
    # 提取小时特征 - 欺诈行为可能在特定时间段更频繁
    train['Transaction_hour'] = (train['TransactionDT'] % (24 * 60 * 60)) // (60 * 60)
    test['Transaction_hour'] = (test['TransactionDT'] % (24 * 60 * 60)) // (60 * 60)
    
    # 提取星期特征 - 周末和工作日可能有不同的欺诈模式
    train['Transaction_day'] = train['TransactionDT_day'] % 7
    test['Transaction_day'] = test['TransactionDT_day'] % 7
    
    # 识别异常时间 - 夜间交易可能更可疑
    train['is_night'] = ((train['Transaction_hour'] <= 6) | (train['Transaction_hour'] >= 22)).astype(int)
    test['is_night'] = ((test['Transaction_hour'] <= 6) | (test['Transaction_hour'] >= 22)).astype(int)
    
    return train, test

# 优化的缺失值处理
def handle_missing_values(train, test):
    print('Handling missing values...')
    
    # 识别分类特征和数值特征
    cat_cols = [col for col in train.columns if train[col].dtype == 'object']
    num_cols = [col for col in train.columns if col not in cat_cols and col != 'isFraud']
    
    # 基于特征类型处理缺失值
    for col in num_cols:
        if col in test.columns:
            # 使用-999填充数值特征，保留缺失信息
            train[col] = train[col].fillna(-999)
            test[col] = test[col].fillna(-999)
        else:
            train[col] = train[col].fillna(-999)
    
    for col in cat_cols:
        if col in test.columns:
            # 使用'missing'填充分类特征
            train[col] = train[col].fillna('missing')
            test[col] = test[col].fillna('missing')
        else:
            train[col] = train[col].fillna('missing')
    
    # 计算缺失值比例特征 - 缺失值模式可能与欺诈相关
    train['missing_count'] = train.isnull().sum(axis=1)
    test['missing_count'] = test.isnull().sum(axis=1)
    
    return train, test

# 优化的分类特征编码
def encode_categorical_features(train, test):
    print('Encoding categorical features...')
    
    # 识别需要编码的分类特征
    # 排除高基数特征，避免维度灾难
    cat_cols = [col for col in train.columns 
                if train[col].dtype == 'object' 
                and col in test.columns 
                and train[col].nunique() < 1000]
    
    # 频率编码 - 保留类别分布信息
    for col in cat_cols:
        freq_encoding = train[col].value_counts(normalize=True)
        train[col+'_freq'] = train[col].map(freq_encoding)
        test[col+'_freq'] = test[col].map(freq_encoding).fillna(0)
    
    # 标签编码
    le = LabelEncoder()
    for col in cat_cols:
        # 合并训练集和测试集进行编码，确保类别覆盖完整
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))
    
    # 高基数分类特征处理
    high_card_cols = ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
    for col in high_card_cols:
        if col in train.columns and col in test.columns:
            # 目标编码（平滑处理）
            target_mean = train.groupby(col)['isFraud'].mean()
            target_count = train.groupby(col)['isFraud'].count()
            smoothing = 100
            
            # 计算平滑后的目标编码
            smooth_mean = ((target_mean * target_count) + (train['isFraud'].mean() * smoothing)) / (target_count + smoothing)
            
            train[col+'_target'] = train[col].map(smooth_mean).fillna(train['isFraud'].mean())
            test[col+'_target'] = test[col].map(smooth_mean).fillna(train['isFraud'].mean())
    
    return train, test

# 优化的特征工程
def feature_engineering(train, test):
    print('Feature engineering...')
    
    # 交易金额相关特征
    train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
    train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
    train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
    train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')
    
    test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
    test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
    test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
    test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')
    
    # 填充可能的NaN值
    train[['TransactionAmt_to_mean_card1', 'TransactionAmt_to_mean_card4', 
          'TransactionAmt_to_std_card1', 'TransactionAmt_to_std_card4']] = train[['TransactionAmt_to_mean_card1', 
          'TransactionAmt_to_mean_card4', 'TransactionAmt_to_std_card1', 
          'TransactionAmt_to_std_card4']].fillna(0)
    
    test[['TransactionAmt_to_mean_card1', 'TransactionAmt_to_mean_card4', 
          'TransactionAmt_to_std_card1', 'TransactionAmt_to_std_card4']] = test[['TransactionAmt_to_mean_card1', 
          'TransactionAmt_to_mean_card4', 'TransactionAmt_to_std_card1', 
          'TransactionAmt_to_std_card4']].fillna(0)
    
    # Vesta特征处理 - 使用PCA降维
    v_cols = [col for col in train.columns if col.startswith('V')]
    if len(v_cols) > 0:
        print(f"Applying PCA to {len(v_cols)} V features...")
        
        # 只处理训练集和测试集共有的V列
        common_v_cols = [col for col in v_cols if col in test.columns]
        
        # 填充缺失值
        train[common_v_cols] = train[common_v_cols].fillna(-999)
        test[common_v_cols] = test[common_v_cols].fillna(-999)
        
        # 标准化
        scaler = StandardScaler()
        train_v_scaled = scaler.fit_transform(train[common_v_cols])
        test_v_scaled = scaler.transform(test[common_v_cols])
        
        # PCA降维 - 保留90%的方差
        pca = PCA(n_components=0.90, random_state=42)
        train_pca = pca.fit_transform(train_v_scaled)
        test_pca = pca.transform(test_v_scaled)
        
        print(f"PCA reduced {len(common_v_cols)} features to {train_pca.shape[1]} features")
        
        # 添加PCA特征
        for i in range(train_pca.shape[1]):
            train[f'V_PCA_{i}'] = train_pca[:, i]
            test[f'V_PCA_{i}'] = test_pca[:, i]
        
        # 删除原始V列
        train.drop(common_v_cols, axis=1, inplace=True)
        test.drop(common_v_cols, axis=1, inplace=True)
    
    # 卡片相关特征组合
    for feature in ['card1','card2','card3','card5']:
        train[f'{feature}_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
        test[f'{feature}_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    
    # 时间间隔特征处理
    d_cols = [f'D{i}' for i in range(1, 16)]
    for col in d_cols:
        if col in train.columns and col in test.columns:
            # 处理无穷大值
            train[col] = train[col].replace([np.inf, -np.inf], np.nan)
            test[col] = test[col].replace([np.inf, -np.inf], np.nan)
            
            # 计算与交易金额的关系
            train[f'{col}_to_TransactionAmt'] = train[col] / train['TransactionAmt']
            test[f'{col}_to_TransactionAmt'] = test[col] / test['TransactionAmt']
    
    # 识别特征处理
    id_cols = [col for col in train.columns if col.startswith('id_') or col in ['DeviceType', 'DeviceInfo']]
    for col in id_cols:
        if col in train.columns and col in test.columns:
            # 计算每个ID的交易频率
            train[f'{col}_count'] = train[col].map(pd.concat([train[col], test[col]], ignore_index=True).value_counts(dropna=False))
            test[f'{col}_count'] = test[col].map(pd.concat([train[col], test[col]], ignore_index=True).value_counts(dropna=False))
    
    # 删除冗余列
    redundant_cols = ['TransactionDT']  # 已经提取了有用的时间特征
    train.drop(redundant_cols, axis=1, errors='ignore', inplace=True)
    test.drop(redundant_cols, axis=1, errors='ignore', inplace=True)
    
    return train, test

# 优化的特征标准化
def normalize_features(train, test, target_column='isFraud'):
    print('Normalizing numerical features...')
    
    # 排除目标列和ID列
    exclude_cols = [target_column, 'TransactionID']
    numerical_cols = [col for col in train.columns 
                      if train[col].dtype != 'object' 
                      and col not in exclude_cols]
    
    # 处理无穷值和缺失值
    for df in [train, test]:
        for col in numerical_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
    
    # 标准化特征
    scaler = StandardScaler()
    train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
    test[numerical_cols] = scaler.transform(test[numerical_cols])
    
    print(f'Normalized {len(numerical_cols)} numerical features')
    return train, test, scaler

# 焦点损失函数 - 改进对欺诈样本的关注
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# 改进的注意力模型 - 增强对欺诈特征的关注
class EnhancedAttentionModel(nn.Module):
    def __init__(self, latent_dim, fraud_weight=1.5, temperature=1.0):
        super().__init__()
        self.fraud_weight = fraud_weight  # 欺诈特征权重放大系数
        self.temperature = temperature  # 注意力温度参数
        
        # 特征变换层 - 增加网络容量
        self.fraud_transform = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 128)
        )
        
        self.normal_transform = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64)
        )
        
        self.global_transform = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, 96)
        )
        
        # 注意力门控机制
        self.attention_gate = nn.Sequential(
            nn.Linear(128+64+96, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 输出3个注意力权重
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
    def forward(self, fraud_feat, normal_feat, global_feat):
        # 应用欺诈特征权重放大
        fraud_feat = fraud_feat * self.fraud_weight
        
        # 特征变换
        f = self.fraud_transform(fraud_feat)
        n = self.normal_transform(normal_feat)
        g = self.global_transform(global_feat)
        
        # 计算注意力权重
        concat = torch.cat([f, n, g], dim=1)
        attention_logits = self.attention_gate(concat) / self.temperature
        attention_weights = F.softmax(attention_logits, dim=1)  # [batch_size, 3]
        
        # 加权特征
        weighted_fraud = f * attention_weights[:, 0].unsqueeze(1)
        weighted_normal = n * attention_weights[:, 1].unsqueeze(1)
        weighted_global = g * attention_weights[:, 2].unsqueeze(1)
        
        # 融合特征并分类
        fused_features = weighted_fraud + weighted_normal + weighted_global
        output = self.classifier(fused_features)
        
        # 返回注意力权重用于监控
        return output, attention_weights

# GAN过采样类
class GAN(nn.Module):
    def __init__(self, input_dim, latent_dim=100):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # 生成器
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, input_dim),
            nn.Tanh()
        ).to(device)
        
        # 判别器
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, z):
        z = z.to(device)
        return self.generator(z)

    def get_discriminator(self):
        return self.discriminator

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        ).to(device)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim),
            nn.Tanh()
        ).to(device)
        
    def forward(self, x):
        x = x.to(device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# 欺诈检测模型
class FraudDetectionSystem:
    def __init__(self, input_dim, latent_dim=32, fraud_weight=1.5, device=device):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        
        # 初始化三个自编码器
        self.fraud_ae = Autoencoder(input_dim, latent_dim).to(device)
        self.normal_ae = Autoencoder(input_dim, latent_dim).to(device)
        self.global_ae = Autoencoder(input_dim, latent_dim).to(device)
        
        # 初始化增强的注意力模型
        self.attention_model = EnhancedAttentionModel(latent_dim, fraud_weight=fraud_weight).to(device)
        
    def pretrain(self, X_fraud, X_normal, X_global, epochs=30, batch_size=64, learning_rate=0.001):
        def train_epoch(model, dataloader, optimizer, criterion, device):
            model.train()
            total_loss = 0
            for data in dataloader:
                inputs = data[0].to(device)
                optimizer.zero_grad()
                decoded, _ = model(inputs)
                loss = criterion(decoded, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(dataloader)
        
        criterion = nn.MSELoss().to(self.device)
        
        # 预训练欺诈自编码器
        dataset_fraud = TensorDataset(torch.FloatTensor(X_fraud).to(self.device))
        loader_fraud = DataLoader(dataset_fraud, batch_size=batch_size, shuffle=True)
        optimizer_f = optim.Adam(self.fraud_ae.parameters(), lr=learning_rate)
        print("预训练欺诈自编码器...")
        for epoch in range(epochs):
            loss = train_epoch(self.fraud_ae, loader_fraud, optimizer_f, criterion, self.device)
            print(f'Fraud AE Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}')
        
        # 预训练正常自编码器
        dataset_normal = TensorDataset(torch.FloatTensor(X_normal).to(self.device))
        loader_normal = DataLoader(dataset_normal, batch_size=batch_size, shuffle=True)
        optimizer_n = optim.Adam(self.normal_ae.parameters(), lr=learning_rate)
        print("预训练正常自编码器...")
        for epoch in range(epochs):
            loss = train_epoch(self.normal_ae, loader_normal, optimizer_n, criterion, self.device)
            print(f'Normal AE Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}')
        
        # 预训练全局自编码器
        dataset_global = TensorDataset(torch.FloatTensor(X_global).to(self.device))
        loader_global = DataLoader(dataset_global, batch_size=batch_size, shuffle=True)
        optimizer_g = optim.Adam(self.global_ae.parameters(), lr=learning_rate)
        print("预训练全局自编码器...")
        for epoch in range(epochs):
            loss = train_epoch(self.global_ae, loader_global, optimizer_g, criterion, self.device)
            print(f'Global AE Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}')
    
    def train_attention_model(self, X_train, y_train, epochs=30, batch_size=64, learning_rate=0.001):
        # 计算类别权重
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # 先固定自编码器，单独训练注意力模型
        for p in self.fraud_ae.parameters(): p.requires_grad = False
        for p in self.normal_ae.parameters(): p.requires_grad = False
        for p in self.global_ae.parameters(): p.requires_grad = False
        
        # 使用焦点损失函数
        criterion = FocalLoss(alpha=class_weights[1], gamma=2)
        
        # 创建数据加载器
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 创建优化器
        optimizer = optim.Adam(self.attention_model.parameters(), lr=learning_rate)
        
        print("第一阶段: 训练注意力模型...")
        for epoch in range(epochs//2):
            self.attention_model.train()
            total_loss = 0
            attention_weights_history = []
            
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                
                # 获取编码表示
                fraud_encoded = self.fraud_ae.encoder(inputs)
                normal_encoded = self.normal_ae.encoder(inputs)
                global_encoded = self.global_ae.encoder(inputs)
                
                # 注意力模型预测
                outputs, attention_weights = self.attention_model(fraud_encoded, normal_encoded, global_encoded)
                
                # 保存注意力权重用于监控
                attention_weights_history.append(attention_weights.detach().cpu().numpy())
                
                # 计算损失
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 计算平均注意力权重
            if attention_weights_history:
                avg_attn = np.mean(np.vstack(attention_weights_history), axis=0)
                print(f"Epoch {epoch+1}/{epochs//2} - 注意力权重: 欺诈={avg_attn[0]:.4f}, 正常={avg_attn[1]:.4f}, 全局={avg_attn[2]:.4f}")
            
            print(f'Epoch [{epoch+1}/{epochs//2}], Loss: {total_loss/len(dataloader):.4f}')
        
        # 解锁自编码器，联合训练
        for p in self.fraud_ae.parameters(): p.requires_grad = True
        for p in self.normal_ae.parameters(): p.requires_grad = True
        for p in self.global_ae.parameters(): p.requires_grad = True
        
        # 联合训练所有参数
        optimizer = optim.Adam([
            {'params': self.fraud_ae.parameters()},
            {'params': self.normal_ae.parameters()},
            {'params': self.global_ae.parameters()},
            {'params': self.attention_model.parameters()}
        ], lr=learning_rate/2)  # 降低学习率
        
        print("第二阶段: 联合训练所有模型组件...")
        for epoch in range(epochs//2, epochs):
            self.fraud_ae.train()
            self.normal_ae.train()
            self.global_ae.train()
            self.attention_model.train()
            
            total_loss = 0
            attention_weights_history = []
            
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                
                # 获取编码表示
                fraud_encoded = self.fraud_ae.encoder(inputs)
                normal_encoded = self.normal_ae.encoder(inputs)
                global_encoded = self.global_ae.encoder(inputs)
                
                # 注意力模型预测
                outputs, attention_weights = self.attention_model(fraud_encoded, normal_encoded, global_encoded)
                
                # 保存注意力权重用于监控
                attention_weights_history.append(attention_weights.detach().cpu().numpy())
                
                # 计算损失
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 计算平均注意力权重
            if attention_weights_history:
                avg_attn = np.mean(np.vstack(attention_weights_history), axis=0)
                print(f"Epoch {epoch+1}/{epochs} - 注意力权重: 欺诈={avg_attn[0]:.4f}, 正常={avg_attn[1]:.4f}, 全局={avg_attn[2]:.4f}")
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    def predict(self, X):
        self.fraud_ae.eval()
        self.normal_ae.eval()
        self.global_ae.eval()
        self.attention_model.eval()
        
        X = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            fraud_encoded = self.fraud_ae.encoder(X)
            normal_encoded = self.normal_ae.encoder(X)
            global_encoded = self.global_ae.encoder(X)
            predictions, _ = self.attention_model(fraud_encoded, normal_encoded, global_encoded)
            predictions = torch.sigmoid(predictions)
        return predictions.cpu().numpy()

def main():
    train_trans, test_trans, train_iden, test_iden, sample_sub = read_data()
    train, test = merge_data(train_trans, test_trans, train_iden, test_iden)
    train_only, test_only = check_column_consistency(train, test)
    train, test = process_dates(train, test)
    train, test = handle_missing_values(train, test)
    train, test = encode_categorical_features(train, test)
    train, test = feature_engineering(train, test)
    train, test, scaler = normalize_features(train, test)
    
    # 保存预处理后的数据集
    print("保存预处理后的数据集...")
    train.to_csv('preprocessed_train.csv', index=False)
    test.to_csv('preprocessed_test.csv', index=False)
    
    # 准备训练数据
    X = train.drop(['isFraud', 'TransactionID'], axis=1)
    y = train['isFraud']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"原始训练集: 正样本={y_train.sum()}, 负样本={len(y_train)-y_train.sum()}")
    
    X_fraud = X_train[y_train == 1].values
    X_normal = X_train[y_train == 0].values
    
    # 训练GAN生成欺诈样本
    print("开始训练GAN过采样器...")
    gan = GAN(input_dim=X_fraud.shape[1], latent_dim=64)
    optimizer_g = optim.Adam(gan.generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(gan.discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss().to(device)
    
    for epoch in range(3000):
        # 生成噪声和样本
        noise = torch.randn(X_fraud.shape[0], 64).to(device)
        generated = gan(noise)
        real_data = torch.FloatTensor(X_fraud).to(device)
        
        # 训练判别器
        optimizer_d.zero_grad()
        d_real = gan.discriminator(real_data)
        d_gen = gan.discriminator(generated.detach())
        
        d_loss = criterion(d_real, torch.ones_like(d_real)) + \
                criterion(d_gen, torch.zeros_like(d_gen))
        
        d_loss.backward()
        optimizer_d.step()
        
        # 训练生成器
        optimizer_g.zero_grad()
        g_loss = criterion(gan.discriminator(generated), torch.ones_like(d_real))
        g_loss.backward()
        optimizer_g.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    # 生成额外的欺诈样本
    n_to_generate = len(X_normal)*1.5 - len(X_fraud)
    generated_fraud = gan(torch.randn(n_to_generate, 64).to(device)).detach().cpu().numpy()
    
    print(f"生成了 {len(generated_fraud)} 个欺诈样本")
    
    # 合并数据
    X_train_gan = np.vstack([X_normal, X_fraud, generated_fraud])
    y_train_gan = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_fraud)), np.ones(len(generated_fraud))])
    print(f"过采样后训练集: 正样本={y_train_gan.sum()}, 负样本={len(y_train_gan)-y_train_gan.sum()}")
    
    # 训练欺诈检测模型 - 增加欺诈权重参数
    model = FraudDetectionSystem(input_dim=X_train_gan.shape[1], latent_dim=32, fraud_weight=1.8, device=device)
    
    print("开始预训练自编码器...")
    model.pretrain(
        X_train_gan[y_train_gan == 1],
        X_train_gan[y_train_gan == 0],
        X_train_gan,
        epochs=30
    )
    
    print("开始训练注意力模型...")
    model.train_attention_model(X_train_gan, y_train_gan, epochs=40)
    
    print("\n在验证集上评估模型...")
    y_pred = model.predict(X_val.values)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    print("分类报告:")
    print(classification_report(y_val.values, y_pred_binary))
    
    print("混淆矩阵:")
    print(confusion_matrix(y_val.values, y_pred_binary))
    
    print(f"ROC AUC: {roc_auc_score(y_val.values, y_pred):.4f}")
    
    # 为测试集生成预测
    print("\n为测试集生成预测...")
    X_test = test.drop(['TransactionID'], axis=1)
    test_preds = model.predict(X_test.values)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'TransactionID': test['TransactionID'],
        'isFraud': test_preds[:, 0]
    })
    submission.to_csv('fraud_detection_submission.csv', index=False)
    print("预测结果已保存到 'fraud_detection_submission.csv'")
    
    return model

if __name__ == "__main__":
    model = main()    




