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


!nvidia-smi


%matplotlib inline
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os


df_quant = pd.read_excel(os.path.join("/kaggle/input/widsdatathon2025/","TRAIN_NEW","TRAIN_QUANTITATIVE_METADATA_new.xlsx"))
labels = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
df_meta = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
df_conn_mat = pd.read_csv("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")


pd.DataFrame(df_conn_mat.loc[0].values).value_counts()


len(np.unique(df_conn_mat.loc[0].values[1:]))


%matplotlib inline
# Plot for Training sample
df_conn_mat_test = pd.read_csv("/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")

from PIL import Image

data = np.array(df_conn_mat_test.loc[1][1:19882]-np.mean(np.array(df_conn_mat_test.loc[1][1:19882]).reshape(-1))/len(df_conn_mat_test.loc[1][1:19882]),dtype=np.float32).reshape(141,141)

from matplotlib import pyplot as plt
plt.imshow(data)
plt.show()


# Plot for Training sample

from PIL import Image

data = np.array(df_conn_mat.loc[1][1:19882]-np.mean(np.array(df_conn_mat.loc[1][1:19882]).reshape(-1))/len(df_conn_mat.loc[1][1:19882]),dtype=np.float32).reshape(141,141)

from matplotlib import pyplot as plt
plt.imshow(data)
plt.show()


df_conn_mat[df_conn_mat.isna().any(axis=1)]


df = pd.merge(left=df_conn_mat,right=labels,on="participant_id",how="inner")


df


# ADHD Outcome 1 -> Person with ADHD
# ADHD Outcome 0 -> Person without ADHD
df["ADHD_Outcome"].value_counts()


# 0 = Male
# 1 = Female
df["Sex_F"].value_counts()


'''
Takes almost 8-10 secs (will optimize)
'''

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader,Dataset

seed=14
torch.manual_seed(seed)


from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

class MultiOutcomeDataset(Dataset):

    def __init__(self,root_dir="/kaggle/input/widsdatathon2025/TRAIN_NEW",batching=False,seed=42,transform=None, include_social=False) -> None:
        self.root_dir = root_dir
        self.seed = seed
        self.include_social = include_social
        self.batching = batching
        self.get_datasets()
    
    
    def get_datasets(self) -> None:
        # Excels
            # TRAIN_QUANTITATIVE_METADATA.xlsx
            # TRAINING_SOLUTIONS.xlsx
            # TRAIN_CATEGORICAL_METADATA.xlsx
        
        df_mat = pd.read_csv(os.path.join(self.root_dir,"TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv"))
        df_sols = pd.read_excel(os.path.join(self.root_dir,"TRAINING_SOLUTIONS.xlsx"))

        df_quant = pd.read_excel(os.path.join(self.root_dir,"TRAIN_QUANTITATIVE_METADATA_new.xlsx")).dropna(axis=0)
        df_meta = pd.read_excel(os.path.join(self.root_dir,"TRAIN_CATEGORICAL_METADATA_new.xlsx")).dropna(axis=0)
        

        # In Inner Join left and right can be anything (doesn't matter)
        self.df = pd.merge(left=df_mat,right=df_sols,on="participant_id",how="inner")


  


        if self.include_social:
            # Inner Join for combined dataset (quant and meta)
            self.df = pd.merge(left=self.df,right=df_quant,on="participant_id",how="inner")
            self.df = pd.merge(left=self.df,right=df_meta,on="participant_id",how="inner")


    

    def __len__(self) -> int:
        return len(self.df)
    
    def get_dataloaders(self,batch_size=32) -> DataLoader:
        """
            This function is custom made easily for us to create dataloaders without initialize multiple objects
        """
        if self.batching:
            batch_size = batch_size
        else:
            batch_size = len(self)

        labels = ["ADHD_Outcome","Sex_F"]

        X = self.df.drop(columns=labels,axis=1)
        X = X.drop(columns="participant_id",axis=1)
        y = self.df[labels]

        X_train,X_test, y_train,y_test = train_test_split(X,y, train_size=0.8,shuffle=True,random_state=self.seed)
        
        # Convert into Tensors
        X_train, y_train = torch.from_numpy(np.array(X_train,dtype=np.float32)), torch.from_numpy(np.array(y_train,dtype=np.float32))
        X_test, y_test = torch.from_numpy(np.array(X_test,dtype=np.float32)), torch.from_numpy(np.array(y_test,dtype=np.float32))

        # Create loader for NN
        train_loader = DataLoader(TensorDataset(X_train,y_train),batch_size=batch_size,shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test),batch_size=batch_size,shuffle=True)
        return train_loader, test_loader
    


    def __getitem__(self, index) -> dict:
        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.df.loc[index]
        
        matrix_linear = sample[1:-2] # leave the participant id and the outcomes
        adhd = sample[-2] # For ADHD use -2
        sex = sample[-1] # For sex use -1

        sample = {"matrix_linear":matrix_linear,"adhd":adhd,"sex":sex}

        if self.transform:
            sample = self.transform(sample)

        return sample

dataset = MultiOutcomeDataset(batching=True)
train_loader, test_loader = dataset.get_dataloaders(batch_size=10)


class MultiOutcomeNN(nn.Module):

    def __init__(self, in_f) -> None:
        super(MultiOutcomeNN,self).__init__()
        self.lin1 = nn.Linear(in_features=in_f,out_features=250)
        self.lin2 = nn.Linear(in_features=250,out_features=125)
        self.lin3 = nn.Linear(in_features=125,out_features=75)
        self.lin4 = nn.Linear(in_features=75,out_features=32)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # For ADHD Predictor
        self.lin11 = nn.Linear(in_features=32,out_features=32)
        self.lin21 = nn.Linear(in_features=32, out_features=16)
        self.lin31 = nn.Linear(in_features=16,out_features=8)
        self.lin41 = nn.Linear(in_features=8, out_features=1)


        # For Sex Predictor
        self.lin12 = nn.Linear(in_features=32,out_features=32)
        self.lin22 = nn.Linear(in_features=32, out_features=16)
        self.lin32 = nn.Linear(in_features=16,out_features=8)
        self.lin42 = nn.Linear(in_features=8, out_features=1)
        

    def forward(self, x) -> torch.Tensor:
        # Embedding Computation
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.relu(self.lin4(x))

        # ADHD Head Predictor
        adhd = self.relu(self.lin11(x))
        adhd = self.relu(self.lin21(adhd))
        adhd = self.relu(self.lin31(adhd))
        adhd = self.sigmoid(self.lin41(adhd))

        # Sex Head Predictor
        sex = self.relu(self.lin12(x))
        sex = self.relu(self.lin22(sex))
        sex = self.relu(self.lin32(sex))
        sex = self.sigmoid(self.lin42(sex))

        return adhd, sex



for idx, i in enumerate(train_loader):
    if idx == 1:
        break
    sample = i
sample[0].shape


model = MultiOutcomeNN(in_f=19900)
device=torch.device(0)
model.to(device=device)


adhd_loss = nn.BCELoss()
sex_loss = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


N_EPOCHS = 200
for epoch in range(N_EPOCHS):
    r_loss = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, outputs = data
        

        gold_adhd = outputs[:,0]
        gold_adhd = gold_adhd.view(gold_adhd.shape[0],1).to(device)

        gold_sex = outputs[:,1]
        gold_sex = gold_sex.view(gold_sex.shape[0],1).to(device)
        
        inputs = inputs.to(device)
        
        adhd, sex = model(inputs)
        
        loss1 = adhd_loss(adhd,gold_adhd)
        loss2 = sex_loss(sex,gold_sex)
        total = loss1+loss2
        r_loss += total

        # print(f"Total Loss for Epoch {epoch} and DataLoader {i} is {total}")

        total.backward()
        optimizer.step()
    print(f"Total Loss for Epoch {epoch} is {r_loss}")


# Evaluating

# Cross Validation 
# Plots
# Early Stopping
# weighted loss function
adhd_real = []
adhd_pred = []

sex_real = []
sex_pred = []
with torch.no_grad():
    model.eval()
    for i, data in enumerate(train_loader):
        inputs, outputs = data

        gold_adhd = outputs[:,0].to(device)
        gold_sex = outputs[:,1].to(device)

        adhd, sex = model(inputs.to(device))
        adhd = adhd.view(-1)
        sex = sex.view(-1)

        adhd = (adhd > 0.5).int()
        sex = (sex > 0.5).int()
        
        adhd_real.extend(gold_adhd.detach().tolist())
        adhd_pred.extend(adhd.detach().tolist())

        sex_real.extend(gold_sex.detach().tolist())
        sex_pred.extend(sex.detach().tolist())

from sklearn.metrics import classification_report
print(classification_report(y_true=adhd_real,y_pred=adhd_pred))

print(classification_report(y_true=sex_real,y_pred=sex_pred))



# Evaluating

adhd_real = []
adhd_pred = []

sex_real = []
sex_pred = []
with torch.no_grad():
    model.eval()
    for i, data in enumerate(test_loader):
        inputs, outputs = data

        gold_adhd = outputs[:,0].to(device)
        gold_sex = outputs[:,1].to(device)

        adhd, sex = model(inputs.to(device))
        
        adhd = adhd.view(-1)
        sex = sex.view(-1)

        adhd = (adhd > 0.5).int()
        sex = (sex > 0.5).int()
        
        adhd_real.extend(gold_adhd.detach().tolist())
        adhd_pred.extend(adhd.detach().tolist())

        sex_real.extend(gold_sex.detach().tolist())
        sex_pred.extend(sex.detach().tolist())


from sklearn.metrics import classification_report
print(classification_report(y_true=adhd_real,y_pred=adhd_pred))

print(classification_report(y_true=sex_real,y_pred=sex_pred))



'''
Takes almost 8-10 secs (will optimize)
'''
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader,Dataset

seed=14
torch.manual_seed(seed)


from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

class MultiOutcomeDataset(Dataset):

    def __init__(self,root_dir="/kaggle/input/widsdatathon2025/TRAIN_NEW",batching=False,seed=42,transform=None, include_social=False) -> None:
        self.root_dir = root_dir
        self.seed = seed
        self.include_social = include_social
        self.batching = batching
        self.get_datasets()
    
    
    def get_datasets(self) -> None:
        # Excels
            # TRAIN_QUANTITATIVE_METADATA.xlsx
            # TRAINING_SOLUTIONS.xlsx
            # TRAIN_CATEGORICAL_METADATA.xlsx
        
        df_mat = pd.read_csv(os.path.join(self.root_dir,"TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv"))
        df_sols = pd.read_excel(os.path.join(self.root_dir,"TRAINING_SOLUTIONS.xlsx"))

        df_quant = pd.read_excel(os.path.join(self.root_dir,"TRAIN_QUANTITATIVE_METADATA_new.xlsx")).dropna(axis=0)
        df_meta = pd.read_excel(os.path.join(self.root_dir,"TRAIN_CATEGORICAL_METADATA_new.xlsx")).dropna(axis=0)
        

        # In Inner Join left and right can be anything (doesn't matter)
        self.df = pd.merge(left=df_mat,right=df_sols,on="participant_id",how="inner")


  


        if self.include_social:
            # Inner Join for combined dataset (quant and meta)
            self.df = pd.merge(left=self.df,right=df_quant,on="participant_id",how="inner")
            self.df = pd.merge(left=self.df,right=df_meta,on="participant_id",how="inner")


    

    def __len__(self) -> int:
        return len(self.df)
    
    def get_dataloaders(self,batch_size=32) -> DataLoader:
        """
            This function is custom made easily for us to create dataloaders without initialize multiple objects
        """
        if self.batching:
            batch_size = batch_size
        else:
            batch_size = len(self)

        labels = ["ADHD_Outcome","Sex_F"]

        X = self.df.drop(columns=labels,axis=1)
        X = X.drop(columns="participant_id",axis=1)
        y = self.df[labels]

        X_train,X_test, y_train,y_test = train_test_split(X,y, train_size=0.8,shuffle=True,random_state=self.seed)
        
        # Convert into Tensors
        X_train, y_train = torch.from_numpy(np.array(X_train,dtype=np.float32)), torch.from_numpy(np.array(y_train,dtype=np.float32))
        X_test, y_test = torch.from_numpy(np.array(X_test,dtype=np.float32)), torch.from_numpy(np.array(y_test,dtype=np.float32))

        # Create loader for NN
        train_loader = DataLoader(TensorDataset(X_train,y_train),batch_size=batch_size,shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test),batch_size=batch_size,shuffle=True)
        return train_loader, test_loader
    


    def __getitem__(self, index) -> dict:
        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.df.loc[index]
        
        matrix_linear = sample[1:-2] # leave the participant id and the outcomes
        adhd = sample[-2] # For ADHD use -2
        sex = sample[-1] # For sex use -1

        sample = {"matrix_linear":matrix_linear,"adhd":adhd,"sex":sex}

        if self.transform:
            sample = self.transform(sample)

        return sample

dataset = MultiOutcomeDataset(batching=True)
train_loader, test_loader = dataset.get_dataloaders(batch_size=10)


class MultiOutcomeNN(nn.Module):

    def __init__(self, in_f) -> None:
        super(MultiOutcomeNN,self).__init__()
        self.lin1 = nn.Linear(in_features=in_f,out_features=250)
        self.lin2 = nn.Linear(in_features=250,out_features=125)
        self.lin3 = nn.Linear(in_features=125,out_features=75)
        self.lin4 = nn.Linear(in_features=75,out_features=32)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # For ADHD Predictor
        self.lin11 = nn.Linear(in_features=32,out_features=32)
        self.lin21 = nn.Linear(in_features=32, out_features=16)
        self.lin31 = nn.Linear(in_features=16,out_features=8)
        self.lin41 = nn.Linear(in_features=8, out_features=1)


        # For Sex Predictor
        self.lin12 = nn.Linear(in_features=32,out_features=32)
        self.lin22 = nn.Linear(in_features=32, out_features=16)
        self.lin32 = nn.Linear(in_features=16,out_features=8)
        self.lin42 = nn.Linear(in_features=8, out_features=1)
        

    def forward(self, x) -> torch.Tensor:
        # Embedding Computation
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.relu(self.lin4(x))

        # ADHD Head Predictor
        adhd = self.relu(self.lin11(x))
        adhd = self.relu(self.lin21(adhd))
        adhd = self.relu(self.lin31(adhd))
        adhd = self.sigmoid(self.lin41(adhd))

        # Sex Head Predictor
        sex = self.relu(self.lin12(x))
        sex = self.relu(self.lin22(sex))
        sex = self.relu(self.lin32(sex))
        sex = self.sigmoid(self.lin42(sex))

        return adhd, sex



class TransformerCls(nn.Module):

    def __init__(self, n_layers, attn_dims, n_heads):
        super(TransformerCls,self).__init__()
        self.n_layers = n_layers
        self.attn_dims = attn_dims
        self.n_heads = n_heads
        
        self.model = nn.Transformer(d_model = attn_dims,
                                   num_encoder_layers  = n_layers,
                                   nhead  = n_heads,
                                   )
        self.encoder = nn.Linear(1, attn_dims)

        self.cls_head = MultiOutcomeNN(in_f=attn_dims)

    def forward(self, x):
        
        if len(x.shape) == 2:
            x = x.view(x.shape[0],-1,1)

        out = self.encoder(x)
        out = self.model.encoder(out)
        out = torch.mean(out, dim=1)

        out = out.view(out.shape[0],-1)
        out = self.cls_head(out)
        return out

        


attn_dims = 64
attn_layers = 2
attn_heads = 4

model = TransformerCls(n_layers=attn_layers,attn_dims=attn_dims,n_heads=attn_heads)
device=torch.device(0)
model.to(device=device)


for idx, i in enumerate(train_loader):
    if idx == 1:
        break
    sample = i
sample[0].shape


adhd_loss = nn.BCELoss()
sex_loss = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


N_EPOCHS = 100
for epoch in range(N_EPOCHS):
    r_loss = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, outputs = data
        

        gold_adhd = outputs[:,0]
        gold_adhd = gold_adhd.view(gold_adhd.shape[0],1).to(device)

        gold_sex = outputs[:,1]
        gold_sex = gold_sex.view(gold_sex.shape[0],1).to(device)
        
        inputs = inputs.to(device)
        
        adhd, sex = model(inputs)
        
        loss1 = adhd_loss(adhd,gold_adhd)
        loss2 = sex_loss(sex,gold_sex)
        total = loss1+loss2
        r_loss += total

        # print(f"Total Loss for Epoch {epoch} and DataLoader {i} is {total}")

        total.backward()
        optimizer.step()
    print(f"Total Loss for Epoch {epoch} is {r_loss}")


# Evaluating

# Cross Validation 
# Plots
# Early Stopping
# weighted loss function
adhd_real = []
adhd_pred = []

sex_real = []
sex_pred = []
with torch.no_grad():
    model.eval()
    for i, data in enumerate(train_loader):
        inputs, outputs = data

        gold_adhd = outputs[:,0].to(device)
        gold_sex = outputs[:,1].to(device)

        adhd, sex = model(inputs.to(device))
        adhd = adhd.view(-1)
        sex = sex.view(-1)

        adhd = (adhd > 0.5).int()
        sex = (sex > 0.5).int()
        
        adhd_real.extend(gold_adhd.detach().tolist())
        adhd_pred.extend(adhd.detach().tolist())

        sex_real.extend(gold_sex.detach().tolist())
        sex_pred.extend(sex.detach().tolist())

from sklearn.metrics import classification_report
print(classification_report(y_true=adhd_real,y_pred=adhd_pred))

print(classification_report(y_true=sex_real,y_pred=sex_pred))



# Evaluating

adhd_real = []
adhd_pred = []

sex_real = []
sex_pred = []
with torch.no_grad():
    model.eval()
    for i, data in enumerate(test_loader):
        inputs, outputs = data

        gold_adhd = outputs[:,0].to(device)
        gold_sex = outputs[:,1].to(device)

        adhd, sex = model(inputs.to(device))
        
        adhd = adhd.view(-1)
        sex = sex.view(-1)

        adhd = (adhd > 0.5).int()
        sex = (sex > 0.5).int()
        
        adhd_real.extend(gold_adhd.detach().tolist())
        adhd_pred.extend(adhd.detach().tolist())

        sex_real.extend(gold_sex.detach().tolist())
        sex_pred.extend(sex.detach().tolist())


from sklearn.metrics import classification_report
print(classification_report(y_true=adhd_real,y_pred=adhd_pred))

print(classification_report(y_true=sex_real,y_pred=sex_pred))


