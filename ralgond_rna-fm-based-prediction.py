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

# os.listdir("/kaggle/input/rna-fm")




import sys
sys.path.append("/kaggle/input/rna-fm/")

%pip install /kaggle/input/rna-fm/ptflops-0.7.4-py3-none-any.whl
%ls /kaggle/input/rna-fm/RNA-FM_pretrained.pth

import fm

# model, alphabet = fm.downstream.build_rnafm_resnet(type="ss", model_location="/kaggle/input/rna-fm/RNA-FM-ResNet_PDB-All.pth")


import torch
import fm

# Load RNA-FM model
model, alphabet = fm.pretrained.rna_fm_t12(model_location="/kaggle/input/rna-fm/RNA-FM_pretrained.pth")
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data
# data = [
#     ("RNA1", "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU"),
#     ("RNA2", "GGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
#     ("RNA3", "CGAUUCNCGUUCCC--CCGCCUCCA"),
# ]


def gen_seq_label_df(sequence_file_path, label_file_path, with_label=True):
    seq_df = pd.read_csv(sequence_file_path)
    raw = []
    for idx,rows in seq_df.iterrows():
        RNA_name = rows['target_id']
        sequence = rows["sequence"][:1022]
        data = [
            ("RNA", sequence)
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
        # Extract embeddings (on CPU)
        with torch.no_grad():
            # print(sequence, len(sequence))
            results = model(batch_tokens, repr_layers=[12])
        token_embeddings = results["representations"][12].squeeze(dim=0)
        assert len(token_embeddings)-2 == len(sequence)
        for idx, resname in enumerate(sequence, start=1):
            ID = RNA_name + "_" + str(idx)
            raw.append((ID, resname, np.concatenate((token_embeddings[0,:], token_embeddings[idx,:]))))
    
    df = pd.DataFrame(raw, columns=["ID", "resname", "repr"])

    if with_label:
        label_df = pd.read_csv(label_file_path)
        df = df.merge(label_df.drop(columns=['resname', 'resid']), how="left", on=["ID"])
    df.info()

    return df

train_seq_df = gen_seq_label_df("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv", "/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
valid_seq_df = gen_seq_label_df("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv", "/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
test_seq_df  = gen_seq_label_df("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv", None, with_label=False)





def to_X_Y(df):
    X_l = []
    Y_l = []
    for i in range(df.shape[0]):
        X_l.append(df.iloc[i, 2])
        Y_l.append(df.iloc[i, 3:6].to_list())
    return np.array(X_l), np.array(Y_l)


train_seq_df = train_seq_df.dropna()
train_X, train_Y = to_X_Y(train_seq_df)
print("train_X.shape:", train_X.shape, "train_Y.shape:", train_Y.shape, "train_Y.dtype:", train_Y.dtype)

valid_X, valid_Y = to_X_Y(valid_seq_df)
print("valid_X.shape:", valid_X.shape, "valid_Y.shape:", valid_Y.shape)

def to_X(df):
    X_l = []
    for i in range(df.shape[0]):
        X_l.append(df.iloc[i, 2])
    return np.array(X_l)

test_X = to_X(test_seq_df)
print("test_X.shape:", test_X.shape)


# print(train_X)
# print(train_Y)

def find_objects(arr):
  """
  在一个 dtype 为 object 的 NumPy 数组中查找非数值对象。

  参数：
    arr: dtype 为 object 的 NumPy 数组。

  返回：
    包含非数值对象索引的列表。
  """

  object_indices = []
  for index, item in np.ndenumerate(arr):
    if isinstance(item, object) and not isinstance(item, (int, float, np.integer, np.floating)):
      print("item:", item)
      object_indices.append(index)
  return object_indices

print(find_objects(train_Y))
print(train_Y.dtype)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 自定义 Dataset
class RNADataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.Y = torch.tensor(Y, dtype=torch.float32).to(device)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

print(train_X.dtype)
print(train_Y.dtype)
train_Y = np.nan_to_num(train_Y, nan=0.0)

train_dataset = RNADataset(train_X, train_Y)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# valid_dataset = RNADataset(valid_X, valid_Y)
# valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 残差连接 (Residual Connection)
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return x + self.fc(self.norm(x))

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(640*2),
            nn.Linear(640*2, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            ResidualBlock(512),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 3)  # 3D坐标输出
        )

    def forward(self, x):
        return self.net(x)
        
# FCN 网络定义
# class FCN(nn.Module):
#     def __init__(self, input_dim=640, hidden_dim=256, output_dim=3):
#         super(FCN, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.fc(x)



# 训练循环
def train(model, train_dataloader, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        for X_batch, Y_batch in train_dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # 验证集评估
        # model.eval()
        # total_val_loss = 0
        # with torch.no_grad():
        #     for X_val, Y_val in val_dataloader:
        #         X_val, Y_val = X_val.to(device), Y_val.to(device)
        #         val_outputs = model(X_val)
        #         val_loss = criterion(val_outputs, Y_val)
        #         total_val_loss += val_loss.item()
        # avg_val_loss = total_val_loss / len(val_dataloader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")


out_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")[['ID','resname','resid']]

fcn_model_list = []
for i in range(1,6):
    torch.manual_seed(i) #设置 CPU 上的随机种子。
    torch.cuda.manual_seed(i) #设置当前 GPU 上的随机种子。

    fcn_model = FCN().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(fcn_model.parameters(), lr=1e-4)

    # 执行训练
    train(fcn_model, train_dataloader)

    test_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)

    result = fcn_model(test_tensor)

    result = result.cpu().detach().numpy()

    print(result, result.shape)

    col_x, col_y, col_z = f"x_{i}", f"y_{i}", f"z_{i}"
    df = pd.DataFrame(result, columns=[col_x, col_y, col_z])
    out_df = pd.concat([out_df, df], axis=1)

out_df.to_csv("submission.csv", index=False)





