import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from torch.utils import data


# device = cuda
train_data = pd.read_csv("/kaggle/input/california-house-prices/train.csv")
test_data = pd.read_csv("/kaggle/input/california-house-prices/test.csv")
# 生成标签
train_label = train_data['Sold Price']
if 'Sold Price' in train_data.columns:
    train_data = train_data.drop(columns='Sold Price')
#合并特征
all_features = pd.concat((train_data.iloc[:,1:],test_data.iloc[:,1:]))
#特征处理
all_features['Bedrooms'] = pd.to_numeric(all_features['Bedrooms'], errors='coerce')
all_features['Listed On'] = pd.to_datetime(all_features['Listed On'], errors='coerce')
all_features['Listed_Year'] = all_features['Listed On'].dt.year
all_features['Listed_Month'] = all_features['Listed On'].dt.month
all_features = all_features.drop(['Last Sold Price','Listed On'],axis=1)
# for in_object in all_features.dtypes[all_features.dtypes=='object'].index:
#     print(in_object.ljust(20),len(all_features[in_object].unique()))
numerical_features = all_features.dtypes[all_features.dtypes!='object'].index
large_vel_cols = ['Lot', 'Total interior livable area', 'Tax assessed value', 'Annual tax amount', 'Listed Price']
all_features[large_vel_cols] = all_features[large_vel_cols].fillna(0)
for c in large_vel_cols:
    all_features[c] = np.log(all_features[c]+1)

#归一化
for col in numerical_features:
    mean = all_features.loc[:train_data.shape[0]-1, col].mean()
    std = all_features.loc[:train_data.shape[0]-1, col].std()
    all_features[col] = (all_features[col] - mean) / std
#处理缺失值
all_features[numerical_features] = all_features[numerical_features].fillna(0)
#处理非数字数据，采用onehot编码
all_features = all_features[list(numerical_features) + ['Type']]
all_features = pd.get_dummies(all_features,dummy_na = True,dtype = int)


#切分训练集和测试集数据
n_train = train_data.shape[0]
X_train = torch.tensor(all_features.iloc[:n_train,:].values,dtype = torch.float32)
y = torch.tensor(train_label.values,dtype = torch.float32).reshape(-1,1)
# y_mean = y.mean()
# y_std = y.std()
# y = (y-y.mean())/y.std()
X_test = torch.tensor(all_features.iloc[n_train:,:].values,dtype = torch.float32)


all_features.head()


print(all_features.shape,X_train.shape,y.shape,X_test.shape)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#模型定义和实例化
class MLP(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        
        self.layer1 = nn.Linear(in_features,128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128,64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64,32)
        self.bn3 = nn.BatchNorm1d(32)
        self.layer4 = nn.Linear(32,1)
    def forward(self,X):
        X = F.relu(self.bn1(self.layer1(X)))
        X = F.relu(self.bn2(self.layer2(X)))
        X = F.relu(self.bn3(self.layer3(X)))
        X = self.layer4(X)
        return X
in_features = X_train.shape[1]
net = MLP(in_features).to(device)
#损失定义
loss = nn.MSELoss()
#数据加载
def load_array(data_arrays,batch_size,is_Train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_Train)


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls,test_ls =[],[]
    train_iter = load_array((train_features,train_labels),batch_size)
    optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate,weight_decay=weight_decay)
    for epoch in range(1,num_epochs+1):
        net.train()
        for X,y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
        train_ls.append(l.item())
        net.eval()
        with torch.no_grad():
            train_ls.append(loss(net(train_features.to(device)),train_labels.to(device)).item())
        if test_labels is not None:
            net.eval()
            with torch.no_grad():
                test_ls.append(loss(net(test_features.to(device)),test_labels.to(device)).item())
        print(f'Epochs:{epoch}, train loss:{train_ls[-1]}')
        if test_labels is not None:
            print(f'Epochs:{epoch}, test loss:{test_ls[-1]}')
    return train_ls,test_ls


num_epochs, lr, weight_decay, batch_size = 500, 0.05, 0.05, 256
train_ls,test_ls = train(net,X_train,y,X_test,None,num_epochs, lr, weight_decay, batch_size)



net.to('cpu')
# preds = net(X_test).detach().numpy() * y_std.item() + y_mean.item()
preds = net(X_test).detach().numpy()
# 将其重新格式化以导出到Kaggle
submission = pd.DataFrame()

submission['Id'] = test_data['Id']  # 或者你数据中能唯一定位每行的字段
submission['Sold Price'] = preds.reshape(-1)  # 将预测结果 flatten
submission.to_csv('submission.csv', index=False)


