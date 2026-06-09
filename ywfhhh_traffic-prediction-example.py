import torch
import os
import pandas as pd
import math
from sklearn import preprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import tqdm
dataset = 'Florida'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_his = 6
n_pred = 1
epochs = 120
batch_size = 1024


def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, 1, n_pred, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i, :, :, :] = data[tail: tail + n_pred].reshape(1, n_pred, n_vertex)

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)


dataset_path = '/kaggle/input/my-traffic-predictmy'
dataset_path = os.path.join(dataset_path, dataset)
if dataset != 'Florida':
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
else:
    data_col = 3648
val_and_test_rate = 0.15
len_val = int(math.floor(data_col * val_and_test_rate))
len_test = int(math.floor(data_col * val_and_test_rate))
len_train = int(data_col - len_val - len_test)
dataset_path = ''
dataset_path = os.path.join(dataset_path, dataset)
if dataset != 'Florida':
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))
else:
    vel = np.load('/kaggle/input/my-traffic-predictmy/poi_hour20190601_20191030.npy')
train = vel[: len_train]
val = vel[len_train: len_train + len_val]
test = vel[len_train + len_val:]
zscore = preprocessing.StandardScaler()
train = zscore.fit_transform(train)# train max=55693,min=0 -> train max=8.65,min=-1.49
val = zscore.transform(val)# train max=55693,min=0 -> train max=11,min=-1
test = zscore.transform(test)# train max=55693,min=0 -> train max=7,min=-1


train.shape,val.shape,test.shape


x_train, y_train = data_transform(train, n_his, n_pred, device)
x_val, y_val = data_transform(val, n_his, n_pred, device)
x_test, y_test = data_transform(test, n_his, n_pred, device)
train_data = utils.data.TensorDataset(x_train, y_train)
train_iter = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
val_data = utils.data.TensorDataset(x_val, y_val)
val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
test_data = utils.data.TensorDataset(x_test, y_test)
test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


x_train.shape, y_train.shape #[样本数, 流量值，时间步，地点]
num_nodes = x_train.shape[-1]


adj = np.load(f'/kaggle/input/my-traffic-predictmy/{dataset}_adj.npy')
adj = torch.from_numpy(adj).to(device)
adj.shape, type(adj)


import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, adj):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_feats, out_feats))
        self.register_buffer('adj', self.normalize_adj(adj))

    def normalize_adj(self, adj):
        # adj = torch.tensor(adj, dtype=torch.float32)
        # print(adj)
        # I = torch.eye(adj.size(0), device=adj.device)
        # adj = adj + I
        # D_inv_sqrt = torch.diag(1.0 / torch.sqrt(adj.sum(1)))
        # print(D_inv_sqrt)
        return adj

    def forward(self, x):
        # x: [batch, num_nodes, in_feats]
        support = torch.matmul(x, self.weight)
        out = torch.matmul(self.adj, support)
        return out
class ForLoopGCN(nn.Module):
    def __init__(self, adj, in_feats=1, hidden_feats=16, out_feats=1, n_his=6, num_nodes=67):
        super(ForLoopGCN, self).__init__()
        self.n_his = n_his
        self.num_nodes = num_nodes
        self.gcn = GCNLayer(in_feats, hidden_feats, adj)
        self.linear = nn.Linear(n_his * hidden_feats, out_feats)

    def forward(self, x):
        # print(x.shape)
        # x: [batch, 1, n_his, num_nodes]
        batch_size = x.size(0)
        outputs = []

        for t in range(self.n_his):
            xt = x[:, :, t, :]         # [B, 1, N]
            # print(xt.shape)
            xt = xt.permute(0, 2, 1)  # [B, N, 1]
            out = F.relu(self.gcn(xt))          # [B, N, hidden_feats]
            outputs.append(out)

        # Concatenate along time: [B, N, n_his * hidden_feats]
        out = torch.cat(outputs, dim=-1)

        # Final prediction: [B, N, out_feats]
        out = self.linear(out)

        # Reshape to [B, 1, 1, N]
        out = out.permute(0, 2, 1).unsqueeze(2)
        return out  # [B, 1, 1, N]


# 假设 adj 是 [67, 67] 的 numpy 数组
model = ForLoopGCN(adj=adj, n_his=6, num_nodes=67).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.L1Loss() #MAE 损失函数


def train(loss, optimizer, model, train_iter, val_iter):
    for epoch in range(epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            # print(x.shape,y.shape)
            optimizer.zero_grad()
            y_pred = model(x).view(len(x), 1, -1)  #9.5605e-03, -1.0963e-01,  1.4936e-01,
            # print(y_pred)
            l = loss(y_pred, y.view(len(x), 1,  -1)) # l = 0.3138
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        val_loss = val(model, val_iter)
        # val_loss = utility.evaluate_model(model, loss, val_iter)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

@torch.no_grad()
def val(model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), 1, -1)
        # y_pred = model(x).reshape(len(x), -1)
        l = loss_fn(y_pred, y) #每一批次的数据计算 MAE
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)
@torch.no_grad() 
def test(zscore, model, test_iter):
    model.eval()
    mae = []
    all_preds = []
    all_labels = []
    for x, y in test_iter:
        batch,  channels, time, num_node = y.shape
        y = y.cpu().numpy().reshape(batch * time, channels*num_node)
        y_pred = model(x).cpu().numpy().reshape(batch*time, channels*num_node)
        y = zscore.inverse_transform(y).reshape(-1)
        y_pred = zscore.inverse_transform(y_pred).reshape(-1)
        all_preds.append(y_pred.reshape(batch, num_node))
        all_labels.append(y.reshape(batch, num_node))
        d = np.abs(y - y_pred)
        mae += d.tolist()
    # 拼接所有批次预测结果
    all_preds = np.concatenate(all_preds, axis=0)  # [total_samples, 67]
    all_labels = np.concatenate(all_labels, axis=0)
    # 构建 DataFrame
    preds_df = pd.DataFrame({
        'id': np.arange(all_preds.shape[0]),
        'traffic': [','.join(map(lambda x: f'{x:.6f}', row)) for row in all_preds]
    })
    labels_df = pd.DataFrame({
        'id': np.arange(all_labels.shape[0]),
        'Usage': 'Public',
        'traffic': [','.join(map(lambda x: f'{x:.6f}', row)) for row in all_labels]
    })
    # labels_df.to_csv('/home/yiwenfeng/my_projects/Graph_Neural_Network/STGCN/Florida_Solution.csv', index=False)
    # 保存为 submission.csv
    preds_df.to_csv('submission.csv', index=False)
    print(f"✅ Submission saved")
    return np.array(mae).mean() 


train(loss_fn, optimizer, model, train_iter, val_iter)


test(zscore, model, test_iter)




