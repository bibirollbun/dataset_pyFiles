!pip install torch_geometric


import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
import pandas as pd
train_dataset = torch.load('/kaggle/input/graph-classification-mutag/ENZYMES_train.pt',weights_only=False)
test_dataset = torch.load('/kaggle/input/graph-classification-mutag/ENZYMES_test.pt',weights_only=False)
print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义GCN图分类模型
class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 6)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        # 1. 节点特征学习
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        
        # 2. 图级池化
        x = global_max_pool(x, batch)
        
        # 3. 分类器
        x = self.lin(x)
        return x


# 创建模型和优化器
model = GCNGraphClassifier(hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


# 训练函数
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_dataset)

# 测试函数
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# 测试函数
def get_prediction(loader):
    model.eval()
    all_preds = []
    for data in loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            all_preds.append(pred)
    return torch.cat(all_preds,dim=0)


# 训练循环
# train_losses = []
# train_accs = []
for epoch in range(1, 11):
    loss = train()
    train_acc = test(train_loader)
    # train_losses.append(loss)
    # train_accs.append(train_acc)
    # test_accs.append(test_acc)
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train Acc: {train_acc:.4f}')


test_pred = get_prediction(test_loader)
# 保存测试结果集
test_ids = list(range(len(test_pred)))  # 或者使用你的测试节点实际ID列表

# test_pred 可能是 Tensor，转换为 numpy 数组并转成 Python 列表
test_pred_list = test_pred.cpu().numpy().tolist()

# 创建 DataFrame
submission_df = pd.DataFrame({
    'Id': test_ids,
    'label': test_pred_list
})

# 保存为 CSV，不带索引列
submission_df.to_csv('submission.csv', index=False)
print("✅ submission.csv 文件已保存")

