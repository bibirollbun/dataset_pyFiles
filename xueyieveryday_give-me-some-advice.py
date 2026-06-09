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


train_data = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
train_data.head()


training_data = train_data.iloc[:, 1:-1]
labels = train_data.iloc[:, -1]
real_train_data = pd.get_dummies(training_data)


real_train_data = real_train_data.astype(float)
import torch
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
real_train_data[num_cols] = pd.DataFrame(scaler.fit_transform(real_train_data[num_cols]))
features_tensor = torch.tensor(real_train_data.values, dtype=torch.float32)
labels = torch.tensor(labels.values, dtype=torch.float32)
labels = labels.float()
labels = labels.unsqueeze(1)

test_features = test_data.iloc[:, 1:]  
test_dummies = pd.get_dummies(test_features)
test_dummies = test_dummies.astype(float)  
test_dummies[num_cols] = pd.DataFrame(scaler.transform(test_dummies[num_cols]))

test_tensor = torch.tensor(test_dummies.values, dtype=torch.float32)


from torch import nn
!pip install --force-reinstall --no-deps d2l==0.17.5
from d2l import torch as d2l
from torch.optim.lr_scheduler import StepLR
in_features = features_tensor.shape[1]
neg_count = (labels == 0).sum().item()
pos_count = (labels == 1).sum().item()
pos_weight = torch.tensor([4.0])
loss = nn.BCEWithLogitsLoss(pos_weight)

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops.layers.torch import Rearrange  # 需要安装：pip install einops

class FTTransformer(nn.Module):
    def __init__(self, 
                 num_features,          # 输入特征的数量（如表格数据的列数）
                 dim_out=1,            # 输出维度（回归任务为1，分类任务为类别数）
                 dim_hidden=64,        # 特征嵌入的维度
                 num_layers=3,         # Transformer编码器层数
                 n_head=4,             # 注意力头数
                 attn_dropout=0.1,     # 注意力层的Dropout
                 ff_dropout=0.1        # FeedForward层的Dropout
                ):
        super().__init__()
        
        # ========== 1. 特征标记化层 ==========
        self.feature_tokenizer = nn.Sequential(
            # 将每个标量特征转换为dim_hidden维向量
            Rearrange('b n -> b n 1'),          # [batch, num_features] -> [batch, num_features, 1]
            nn.Linear(1, dim_hidden)            # 对每个特征独立映射 [..., 1] -> [..., dim_hidden]
        )  # 输出形状: [batch, num_features, dim_hidden]
        
        # ========== 2. [CLS]标记和位置编码 ==========
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden))  # 可学习的[CLS]标记
        self.pos_embedding = nn.Parameter(torch.randn(1, num_features + 1, dim_hidden))
        
        # ========== 3. Transformer编码器 ==========
        encoder_layers = TransformerEncoderLayer(
            d_model=dim_hidden,
            nhead=n_head,
            dim_feedforward=4*dim_hidden,
            dropout=attn_dropout,
            activation='gelu',
            batch_first=True  # 输入输出为[batch, seq, feature]
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # ========== 4. 输出层 ==========
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_out)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """ Xavier均匀初始化 """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # CLS标记特殊初始化
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """ 
        输入: 
            x - [batch_size, num_features] 
        输出: 
            [batch_size, dim_out] 
        """
        # 1. 特征标记化
        tokens = self.feature_tokenizer(x)  # [batch, num_features, dim_hidden]
        
        # 2. 添加[CLS]标记
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [batch, 1, dim_hidden]
        tokens = torch.cat([cls_tokens, tokens], dim=1)        # [batch, num_features+1, dim_hidden]
        
        # 3. 添加位置编码
        tokens += self.pos_embedding
        
        # 4. 通过Transformer
        encoded = self.transformer(tokens)  # [batch, num_features+1, dim_hidden]
        
        # 5. 使用[CLS]标记作为全局表示
        cls_encoded = encoded[:, 0, :]      # [batch, dim_hidden]
        
        # 6. 输出预测
        return self.to_out(cls_encoded)     # [batch, dim_out]


def train(net, features_tensor, labels, num_epochs, lr, weight_decay, batch_size):
    train_ls = []
    train_iter = d2l.load_array((features_tensor, labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    print("Starting training...")
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            
        epoch_loss = loss(net(features_tensor), labels).item()
        train_ls.append(epoch_loss)
        
        # Print the loss for this epoch
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        scheduler.step()
    
    print("Training completed!")
    return train_ls


num_epochs = 40      
lr = 0.00001           
weight_decay = 0.01 
batch_size = 64     



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

best_auc = 0
best_model = None
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 假设 feature_tensor 是形状为 (n_samples, n_features) 的 PyTorch 张量
# labels 是形状为 (n_samples,) 的 PyTorch 张量
y_probs = torch.zeros(len(test_tensor))  # 假设 test_tensor 也是 PyTorch 张量
models = []

# 将 PyTorch 张量转换为 NumPy 数组以用于 KFold.split()
feature_np = features_tensor.numpy() if isinstance(features_tensor, torch.Tensor) else features_tensor
labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

for fold, (train_idx, val_idx) in enumerate(kf.split(feature_np, labels_np)):
    print(f'Training fold {fold+1}/{n_splits} >>>')
    
    # 使用 PyTorch 张量的索引方式
    X_train, y_train = features_tensor[train_idx], labels[train_idx]
    X_val, y_val = features_tensor[val_idx], labels[val_idx]

    net = FTTransformer(
    num_features=in_features,
    dim_out=1,          # 二分类输出1维（用sigmoid激活）
    dim_hidden=64,
    num_layers=4
    )
    
    train_losses = train(
        net=net,
        features_tensor=X_train,  # 已经是张量，无需转换
        labels=y_train,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size
    )
    models.append(net)

    # 验证集评估
    with torch.no_grad():
        y_val_pred = net(X_val).squeeze()
        val_auc = roc_auc_score(y_val.numpy(), y_val_pred.numpy())
        if val_auc > best_auc:
            best_auc, best_model = val_auc, net
        print(f'Fold {fold+1} Val AUC:', val_auc)

    # 测试集预测（累积概率）
    with torch.no_grad():
        y_pred = net(test_tensor).squeeze()
        y_probs += y_pred / n_splits

if best_model:
    torch.save(best_model.state_dict(), 'best_model.pth')
    print(f'Best Val AUC: {best_auc:.4f}')
import matplotlib.pyplot as plt

print("Training Loss:", train_losses)

plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


outputs = best_model(test_tensor)


probabilities = torch.sigmoid(outputs).detach().numpy().squeeze()
probabilities = np.where(probabilities < 0.01, 0, 
                        np.where(probabilities > 0.99, 1, probabilities))
submission = pd.DataFrame({
    "id": test_data["id"], 
    "y": probabilities
})

submission.to_csv("submission5.csv", index=False)

