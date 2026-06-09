!pip install torch_geometric


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
import pandas as pd
data = torch.load('/kaggle/input/pubmed/data.pt')
data = data[0]
X = data['x']
Y = data['y']
edge_index = data['edge_index']
train_mask = data['train_mask']
val_mask = data['val_mask']
test_mask = data['test_mask']


# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(500, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 3)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)



# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(X, edge_index)
    loss = criterion(out[train_mask], Y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()   

def get_output(model):
    model.eval()
    with torch.no_grad():
        out = model(X, edge_index)
        pred = out.argmax(dim=1)
        test_pred = pred[test_mask]
    return test_pred


# 创建模型和优化器
model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()



# train_losses = []
# val_accs = []

# 训练 200 轮
for epoch in range(1, 101):
    # 每轮训练
    loss = train()
    # train_losses.append(loss)
    # 每轮验证
    out = model(X, edge_index)
    pred = out.argmax(dim=1)
    val_correct = pred[val_mask] == Y[val_mask]
    val_acc = int(val_correct.sum()) / int(val_mask.sum())
    # val_accs.append(val_acc)

    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} ')

# 获取预测结果
test_pred = get_output(model)



# train_losses = []
# val_accs = []

# 训练 200 轮
for epoch in range(1, 201):
    # 每轮训练
    loss = train()
    # train_losses.append(loss)
    # 每轮验证
    out = model(X, edge_index)
    pred = out.argmax(dim=1)
    val_correct = pred[val_mask] == Y[val_mask]
    val_acc = int(val_correct.sum()) / int(val_mask.sum())
    # val_accs.append(val_acc)

    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} ')

# 获取预测结果
test_pred = get_output(model)


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




