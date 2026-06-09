# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import pointbiserialr

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)


# ==================== 数据描述 ====================
# 加载数据
train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')

# 合并demographics数据
train = train.merge(train_demo, on='subject', how='left')
test = test.merge(test_demo, on='subject', how='left')

# 查看数据基本信息
print("训练集形状:", train.shape)
print("测试集形状:", test.shape)
print("训练demo形状:", train_demo.shape)
print("测试demo形状:", test_demo.shape)
print("\n训练集列名:", len(train.columns.tolist()))
print("测试集列名:", len(test.columns.tolist()))
print("\n训练集前5行:")
print(train.head())


# ==================== 描述性分析 ====================
# 手势类型分布
plt.figure(figsize=(14, 6))
gesture_counts = train['gesture'].value_counts()
plt.bar(gesture_counts.index, gesture_counts.values)
plt.title('Gesture Type Distribution in Training Set')
plt.xlabel('Gesture Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
print(f"\n手势类别分布:\n{gesture_counts}")


# 数值特征分布
numeric_features = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, feature in enumerate(numeric_features):
    axes[i].hist(train[feature].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

# 移除多余的子图
for i in range(len(numeric_features), 8):
    fig.delaxes(axes[i])
plt.tight_layout()
plt.show()


# demographic特征分布
demo_features = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(demo_features):
    axes[i].hist(train[feature].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()


# ==================== 数据清洗 ====================
# 处理TOF传感器的-1值
tof_cols = [col for col in train.columns if col.startswith('tof_')]
for col in tof_cols:
    train[col] = train[col].replace(-1, np.nan)
    test[col] = test[col].replace(-1, np.nan)

# 填充缺失值 - 只使用训练集统计量
numeric_cols = train.select_dtypes(include=np.number).columns.drop(['gesture'], errors='ignore')
for col in numeric_cols:
    if train[col].isnull().any():
        mean_val = train[col].mean()
        train[col] = train[col].fillna(mean_val)
        if col in test.columns:
            # 使用训练集的统计量填充测试集，避免数据泄露
            test[col] = test[col].fillna(mean_val)


# 数值型列
numeric_cols = train.select_dtypes(include=np.number).columns
# 非数值型列
non_numeric_cols = train.select_dtypes(exclude=np.number).columns

print("数值型列数量：", len(numeric_cols))
print("非数值型列数量：", len(non_numeric_cols))
print("\n非数值型列名称：", list(non_numeric_cols))


len(numeric_cols)


# ==================== 特征变换 ====================
# 四元数归一化
quat_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
for df in [train, test]:
    quat_norm = np.sqrt(df[quat_cols].pow(2).sum(axis=1))
    # 避免除以零
    quat_norm[quat_norm == 0] = 1e-10
    df[quat_cols] = df[quat_cols].div(quat_norm.values.reshape(-1,1), axis=0)


# ==================== 数据扩增 ====================
# 时间特征
for df in [train, test]:
    df['sequence_counter_norm'] = df.groupby('sequence_id')['sequence_counter'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10))  # 避免除以零

# 增强的特征工程
def create_advanced_features(df):
    # 基础特征
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_mag'] = np.sqrt(df['rot_x']**2 + df['rot_y']**2 + df['rot_z']**2)
    
    # 角速度特征 (使用四元数差分)
    df['acc_diff'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_diff'] = df.groupby('sequence_id')['rot_mag'].diff().fillna(0)
    # TOF特征统计
    tof_features = []
    for i in range(1, 6):
        sensor_cols = [col for col in tof_cols if col.startswith(f'tof_{i}_')]
        if sensor_cols:
            df[f'tof_{i}_mean'] = df[sensor_cols].mean(axis=1)
            df[f'tof_{i}_std'] = df[sensor_cols].std(axis=1)
            df[f'tof_{i}_max'] = df[sensor_cols].max(axis=1)
            df[f'tof_{i}_min'] = df[sensor_cols].min(axis=1)
            tof_features.extend([f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_max', f'tof_{i}_min'])
    
    # 温度传感器特征
    thm_cols = [f'thm_{i}' for i in range(1, 6)]
    df['thm_mean'] = df[thm_cols].mean(axis=1)
    df['thm_std'] = df[thm_cols].std(axis=1)
    df['thm_max'] = df[thm_cols].max(axis=1)
    df['thm_min'] = df[thm_cols].min(axis=1)
    # 序列统计特征
    seq_group = df.groupby('sequence_id')
    df['seq_acc_mean'] = seq_group['acc_mag'].transform('mean')
    df['seq_acc_std'] = seq_group['acc_mag'].transform('std')
    df['seq_rot_mean'] = seq_group['rot_mag'].transform('mean')
    df['seq_rot_std'] = seq_group['rot_mag'].transform('std')
    
    return df, tof_features

train, tof_features = create_advanced_features(train)
test, _ = create_advanced_features(test)



# ==================== 探索性分析 ====================
# 相关性分析热图
plt.figure(figsize=(20, 16))
selected_features = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 
                    'acc_mag', 'rot_mag', 'acc_diff', 'rot_diff', 'thm_mean', 
                    'age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
corr_matrix = train[selected_features].corr()

# 绘制热图
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# 特征与目标变量的相关性分析
target_corr = {}
y_encoded = LabelEncoder().fit_transform(train['gesture'])
for feature in selected_features:
    corr, _ = pointbiserialr(train[feature], y_encoded)
    target_corr[feature] = corr

# 转换为DataFrame并排序
target_corr_df = pd.DataFrame.from_dict(target_corr, orient='index', columns=['Correlation'])
target_corr_df.sort_values('Correlation', ascending=False, inplace=True)

# 绘制相关性条形图
plt.figure(figsize=(12, 8))
sns.barplot(x=target_corr_df['Correlation'], y=target_corr_df.index, palette='viridis')
plt.title('Feature Correlation with Target Variable', fontsize=16)
plt.xlabel('Point-Biserial Correlation Coefficient')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# 类别与特征的箱线图分析
selected_gestures = ['Text on phone', 'Neck - scratch', 'Eyebrow - pull hair', 
                    'Forehead - scratch', 'Wave hello', 'Write name in air']
selected_features_for_boxplot = ['acc_mag', 'rot_mag', 'thm_mean', 'age']

for feature in selected_features_for_boxplot:
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='gesture', y=feature, data=train[train['gesture'].isin(selected_gestures)], 
                palette='Set2')
    plt.title(f'Distribution of {feature} by Gesture Type', fontsize=14)
    plt.xlabel('Gesture Type')
    plt.ylabel(feature)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


len(tof_features)


len(numeric_cols)


# ==================== 特征选择 ====================
# 选择最终特征列
base_features = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
engineered_features = ['acc_mag', 'rot_mag', 'acc_diff', 'rot_diff', 
                       'thm_mean', 'thm_std', 'thm_max', 'thm_min',
                       'seq_acc_mean', 'seq_acc_std', 'seq_rot_mean', 'seq_rot_std',
                       'sequence_counter_norm']
feature_cols = base_features + engineered_features + tof_features

# 特征数量对比
plt.figure(figsize=(10, 6))
bars = plt.bar(['Original features', 'Engineered features'], [len(numeric_cols), len(feature_cols)], 
               color=['skyblue', 'lightgreen'], alpha=0.8)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')
plt.title('Number of Features Comparison')
plt.ylabel('Number of features')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# ==================== 数据集划分 ====================
# 编码目标变量
le_gesture = LabelEncoder()
y = le_gesture.fit_transform(train['gesture'])

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train[feature_cols].values)
X_test_scaled = scaler.transform(test[feature_cols].values)

# 划分训练验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# ==================== 模型构建与训练 ====================
# 定义数据集类
class GestureDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# 改进的神经网络模型
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ImprovedNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.layer4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.layer4(x)))
        x = self.output(x)
        return x

# 定义竞赛指标
def competition_metric(y_true, y_pred):
    """计算竞赛指标：(Binary F1 + Macro F1)/2"""
    binary_f1 = f1_score(np.where(y_true <= 7, 1, 0),
                         np.where(y_pred <= 7, 1, 0),
                         zero_division=0.0)
    macro_f1 = f1_score(np.where(y_true <= 7, y_true, 99),
                        np.where(y_pred <= 7, y_pred, 99),
                        average="macro", 
                        zero_division=0.0)
    return 0.5 * (binary_f1 + macro_f1), binary_f1, macro_f1

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 计算类别权重以处理不平衡问题
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = torch.FloatTensor(class_weights).to(device)

# 创建数据加载器
train_dataset = GestureDataset(X_train, y_train)
val_dataset = GestureDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# 初始化模型
input_dim = X_train.shape[1]
num_classes = len(le_gesture.classes_)
model = ImprovedNeuralNetwork(input_dim, num_classes)
model.to(device)



# 使用加权损失函数处理类别不平衡
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

# 训练模型
best_metric = 0
num_epochs = 50
train_loss_history = []
val_loss_history = []
metric_history = []
binary_f1_history = []
macro_f1_history = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # 验证阶段
    model.eval()
    val_loss = 0
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(batch_y.cpu().numpy())
    
    # 计算指标
    metric_value, binary_f1, macro_f1 = competition_metric(np.array(all_true), np.array(all_preds))
    scheduler.step(metric_value)
    
    # 记录历史
    train_loss_history.append(train_loss/len(train_loader))
    val_loss_history.append(val_loss/len(val_loader))
    metric_history.append(metric_value)
    binary_f1_history.append(binary_f1)
    macro_f1_history.append(macro_f1)
    
    # 保存最佳模型
    if metric_value > best_metric:
        best_metric = metric_value
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Competition Metric: {metric_value:.4f} (Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})')

# ==================== 模型评估 ====================
# 加载最佳模型并评估
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_true.extend(batch_y.cpu().numpy())

metric_value, binary_f1, macro_f1 = competition_metric(np.array(all_true), np.array(all_preds))
print(f"\n最佳模型验证集表现:")
print(f"Competition Metric: {metric_value:.4f}")
print(f"Binary F1: {binary_f1:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

# 绘制训练曲线
plt.figure(figsize=(16, 6))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss', color='blue')
plt.plot(val_loss_history, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# 指标曲线
plt.subplot(1, 2, 2)
plt.plot(metric_history, label='Competition Metric', color='green')
plt.plot(binary_f1_history, label='Binary F1', color='red', linestyle='--')
plt.plot(macro_f1_history, label='Macro F1', color='purple', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation Metrics')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 生成详细评估报告
print("\n详细分类报告:")
print(classification_report(all_true, all_preds, target_names=le_gesture.classes_))

# 绘制混淆矩阵
plt.figure(figsize=(12, 10))
cm = confusion_matrix(all_true, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_gesture.classes_, 
            yticklabels=le_gesture.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ==================== 预测与提交 ====================
# 预测测试集
test_dataset = GestureDataset(X_test_scaled)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

model.eval()
test_pred_encoded = []
with torch.no_grad():
    for batch_X in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        test_pred_encoded.extend(predicted.cpu().numpy())

# 标签解码
test_pred = le_gesture.inverse_transform(test_pred_encoded)

# 创建提交文件
submission_df = test[['sequence_id']].copy()
submission_df['gesture'] = test_pred
submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print("\n提交文件已保存: /kaggle/working/submission.csv")
print(submission_df.head())








