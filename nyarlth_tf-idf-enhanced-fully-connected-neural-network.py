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


sample_submission = pd.read_csv('/kaggle/input/california-house-prices/sample_submission.csv')
train_data = pd.read_csv('/kaggle/input/california-house-prices/train.csv')
test_data = pd.read_csv('/kaggle/input/california-house-prices/test.csv')


sample_submission.shape, train_data.shape, test_data.shape


test_data.head()


train_data.head()


# 提取所有训练测试数据，但不包括房价
# 前 47000个 为训练数据，后 30000 个是测试数据
all_features = pd.concat((train_data.loc[:, train_data.columns != 'Sold Price'], test_data.iloc[:, 1:]))
all_features = all_features.loc[:, all_features.columns != "Id"]
num_feature = all_features.dtypes[all_features.dtypes != "object"].index
all_features.info()





all_features["Address"]


all_features.info()


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt


# 缩小数字特征的大小，置零均值，将缺失数据置零

# num_feature = all_features.dtypes[all_features.dtypes != "object"].index
all_features[num_feature] = all_features[num_feature].apply(lambda x: (x - x.mean())/x.std())
all_features[num_feature] = all_features[num_feature].fillna(0)


def tfidf(words):
    vectorizer = CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(words))
    # del weight
    # weight=tfidf.toarray().sum(axis = 1)
    weight = np.array([tfidf[i,:].sum() for i in range(tfidf.shape[0])])
    return weight


obj_feature = all_features.dtypes[all_features.dtypes == "object"].index
# obj_feature[1]
all_features[obj_feature] = all_features[obj_feature].fillna("")
all_features[obj_feature] = all_features[obj_feature].apply(tfidf)


all_features.info()


include_text_features = all_features#.copy()
only_num_features = all_features[num_feature]




# 转换为张量形式
# 训练集的大小
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,
                              dtype=torch.float32)
print(train_features.shape)
test_features = torch.tensor(all_features[n_train:].values,
                             dtype=torch.float32)
print(test_features.shape)
train_labels = torch.tensor(train_data['Sold Price'].values.reshape(-1, 1),
                            dtype=torch.float32)
n_features = train_features.shape[1]


# 定义模型与损失函数

loss = nn.MSELoss()


def get_net(n_features):
    net = nn.Sequential(
        nn.Linear(n_features, 1),  # 第一层: n_features -> 1
        nn.ReLU(),                   # 激活函数
    )
    return net


# 使用最小对数平方差定义损失函数
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


# 定义数据迭代器
def DataLoader(data_arrays, batch_size, is_train=True):
    
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    
    train_ls, test_ls = [], []
    
    train_iter = DataLoader((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# 交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


train_ls, valid_ls = train(get_net(n_features), *get_k_fold_data(5, 1, train_features, train_labels), 100, 2,
                            0, 64)


plt.plot(train_ls,'r')
plt.plot(valid_ls,'blue')
plt.xlabel("epoch")
plt.ylabel("loss")
valid_ls[-1]


train_ls, valid_ls = train(get_net(n_features), train_features, train_labels, None, None,
                           200, 5,0, 64)


plt.plot(train_ls,'r')
plt.plot(valid_ls,'blue')
plt.xlabel("epoch")
plt.ylabel("loss")
train_ls[-1]


def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net(n_features)
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)

    print(f'train log rmse {float(train_ls[-1]):f}')
    preds = net(test_features).detach().numpy()
    # preds = np.exp(preds)
    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('submission.csv', index=False)

    return submission

submission = train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)


submission.head()

