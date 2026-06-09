# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import statistics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,  Subset, random_split
import torch.optim as optim
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer


train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
train_sequence = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
val_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
val_sequence = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
test_sequence = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")


train_labels["resid"].idxmax()


train_labels["resid"].max()


maxlength=4300


train_labels["resid"].mean()


px.box(train_labels["resid"])


train_sequence.loc[train_sequence["target_id"]=="4V6X_A5"]


one_seq=train_sequence.iloc[(1,1)]


one_seq


rna=train_sequence[["target_id","sequence"]].copy()


rna


rna.dtypes


#PAD_TOKEN = '<PAD>'


#rna.loc['sequence']=rna['sequence'].apply(lambda x:str(x)+' ')


rna['sequence'].iloc[0]


rna


unique_nuc =set()
for nuc in one_seq:
    unique_nuc.add(nuc)


unique_nuc=sorted(unique_nuc)


unique_nuc


one_seq_chars=[char for char in one_seq]


one_seq_chars


char_to_idx ={char: idx+1 for idx, char in enumerate(unique_nuc)}


char_to_idx


char_to_idx[" "] = 0


char_to_idx


def encode_text(text, max_length):
    encoded = [char_to_idx.get(ch, 0) for ch in text]  # Convert chars to IDs
    return encoded[:max_length] + [0] * (max_length - len(encoded))


rna['num_seq']=rna['sequence'].apply(lambda x: encode_text(x, maxlength))


rna['num_seq']=rna['num_seq'].apply(lambda x: np.array(x))


len(rna.iloc[0,2])


#len(rna.iloc[1,3])


gx=train_labels[train_labels['resname']=='G']


gx['x_1'].value_counts()


px.histogram(gx['x_1'])


train_labels.isnull().sum()


train_labels=train_labels.dropna()


train_labels.head()


cols=['x_1','y_1','z_1']
remaining_cols=['ID','resname','resid']
minmax_scaler = MinMaxScaler()
#minmax = minmax_scaler.fit_transform(train_labels[cols])

standard_scaler = StandardScaler()
#data_standardized = standard_scaler.fit_transform(train_labels[cols])

preprocessor = ColumnTransformer(
    transformers=[
        #('std', StandardScaler(), cols),
        ('minmax', MinMaxScaler(), cols),
        ('passthrough', 'passthrough', remaining_cols)
    ]
)

xyz_std=preprocessor.fit_transform(train_labels)

train_labels_norm = pd.DataFrame(xyz_std, columns=cols+remaining_cols)


train_labels_norm


renc=[]
for i,row in rna.iterrows():
    #print(i)
    temp_label=train_labels[train_labels['ID'].str.startswith( row['target_id'])]
    np_temp=temp_label[['x_1','y_1','z_1']].to_numpy()
    padding = np.zeros((4300 - len(np_temp),np_temp.shape[1]), dtype=np_temp.dtype)
    #print(np_temp)
    renc.append(np.concatenate((np_temp, padding)))


renc[0].shape


#renc[0]


rna['encoded'] = renc    


temp_label=train_labels[train_labels['ID'].str.startswith( row['target_id'])]
np_temp=temp_label[['x_1','y_1','z_1']].to_numpy()


#train_labels[train_labels['ID']== '1SCL_A']


np_temp.shape


rna


rna.isnull().sum()


rna.iloc[1,2]


len(rna)


rna.shape[0]


#npchar_index=np.array(list(char_to_idx.keys())).reshape(-1, 1)


#npchar_index


#enc=OneHotEncoder()
#enc.fit_transform(npchar_index)


#enc.categories_


#np_one_seq=np.array(one_seq_chars).reshape(-1,1)


#np_one_seq


#np_one_seq_encoded=enc.transform(np_one_seq)


#np_one_seq_encoded.toarray()


#position = np.arange(4500)[:, np.newaxis]
#div_term = np.exp(np.arange(0, 4, 2) * (-np.log(10000.0) / 4))


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=3, nhead=3, num_layers=3):
        super().__init__()
        self.d_model = d_model
        
        # Encoder
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(MAX_LENGTH, d_model)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        #linear1 = nn.Linear(3, 64)
        
        # Decoder
        self.decoder_layers = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        positions = torch.arange(src.size(1), device=src.device)
        tok_emb = self.token_embed(src) * np.sqrt(self.d_model)
        #print(tok_emb)
        pos_emb = self.pos_embed(positions)
        #print(pos_emb)
        encoded = self.encoder(tok_emb + pos_emb)
        #print(encoded)
        return encoded

    def decode(self, mem):
        tgt=tgt.float()
        batch_size, tgt_seq_len,_ = tgt.size()
        tgt=tgt.reshape(4,-1)
        #print(tgt.shape)
        L1=nn.Linear(4300*3,4300*2)
        L2=nn.Linear(4300*2,4300)
        #print(tgt.dtype)
        tgt2=L1(tgt)
        #print(tgt2.dtype)
        tgt3=L2(tgt2)
        #print(tgt3.shape)

    def forward(self, src):
        enc = self.encode(src)
        #outputs = self.decode(memory, tgt)
        return enc


class SequenceDataset(Dataset):
    def __init__(self, rna, max_length):
        self.max_length = max_length
        self.num_enc= rna['num_seq'].to_numpy()
        self.nuc_enc= rna['encoded'].to_numpy()
        #col_mean = np.nanmean(self.num_enc, axis=0)
        #self.num_enc = np.where(np.isnan(self.num_enc), 0, self.num_enc)
        #col_mean = np.nanmean(self.nuc_enc, axis=0)
        #self.nuc_enc = np.where(np.isnan(self.nuc_enc), 0, self.nuc_enc)
    
    def __len__(self): return rna.shape[0]
        
    def __getitem__(self, idx): return self.num_enc[idx], self.nuc_enc[idx]


rna.shape[0]


num_enc= rna['num_seq'].to_numpy()
nuc_enc= rna['encoded'].to_numpy().reshape(-1,1)


num_enc[0].shape


nuc_enc[0]





nuc_enc[0]


num_enc.shape


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Hyperparameters
MAX_LENGTH = 4300
BATCH_SIZE = 4
VOCAB_SIZE = 5

# Initialize
model = Seq2SeqTransformer(VOCAB_SIZE)
#criterion = nn.CrossEntropyLoss(ignore_index=4)# Ignore padding
criterion = nn.MSELoss()
#criterion = nn.CosineSimilarity()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


model=model.to(device)


dataset = SequenceDataset(rna, MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


batch=next(iter(dataloader))


batch[0].shape


batch[1].shape


batch[0].size(1)


rna_seq_only=pd.DataFrame(rna['sequence'])


rna_seq_only


d_len=len(dataset)


train_ratio = 0.85
validation_ratio = 0.15

# Step 4: Calculate the sizes for each split
dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
validation_size = dataset_size - train_size


train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)


print(f'Total dataset size: {dataset_size}')
print(f'Training dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(validation_dataset)}')


from tqdm import tqdm


torch.cuda.empty_cache()


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


epochs=100
model.train()
for epoch in range(epochs):
    total_loss = 0
    train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
    for src, tgt in train_bar:
        src, tgt=src.to(device), tgt.to(device)
        #src= src.to(device)
        #src = src.float()
        tgt = tgt.float()
        optimizer.zero_grad()
        #print(src)
        #print(tgt.dtype)
        
        # Shift target for teacher forcing
        
        output = model(src)
        #print(output.dtype)
        #print(tgt.dtype)
        #outputscpu=outputs.to('cpu')
        loss = criterion(output, tgt)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')


tgt


#tgt


#loss=criterion(output, tgt)


#loss


#train(model, dataloader, epochs=10)


#tgt_input


test_outs=[]
testlabels=[]
test_loss=[]
for inputs, tar in validation_loader:
    inputs, tar=inputs.to(device), tar.to(device)
    test_outputs = model(inputs)
    output = model(src)
    loss = criterion(output, tgt)
    test_loss.append( loss.item())
    outputscpu=test_outputs.to('cpu')
    nptestout=outputscpu.detach()
    nptestout=nptestout.numpy()
    for ele in nptestout:
        #print(nptestout.shape)
        test_outs.append(ele)


output.shape


len(test_outs)


test_outs[0]


test_outs[0].shape


len(test_loss)

