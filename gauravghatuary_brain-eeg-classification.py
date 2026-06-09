# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Importing libraries

from pathlib import Path
import re

import matplotlib.pyplot as plt

import json


#Setting the seed for np

np.random.seed(42)


#exploring the  files
current_path=Path(os.getcwd())
input_path=Path('/kaggle/input/hms-harmful-brain-activity-classification')

os.listdir(input_path)


#Defing the spectrograms and eegs dir
train_spectrogram_dir=input_path/'train_spectrograms'
train_eegs_dir=input_path/'train_eegs'

test_spectrogram_dir=input_path/'test_spectrograms'
test_eegs_dir=input_path/'test_eegs'


#Checking the train data
train_data=pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
train_data


train_data.columns


train_data.info()


print(f"No of patients:{train_data['patient_id'].nunique()}")


train_data['expert_consensus'].value_counts().plot(kind='bar')


vote_columns=['seizure_vote', 'lpd_vote','gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
train_data[vote_columns].value_counts()



#Calculating total votes
train_data['total_vote']=train_data[vote_columns].sum(axis=1)

#Calculating the normalized vote for each
for col in vote_columns:
    train_data[col+'_n']=train_data[col]/train_data['total_vote']

train_data



vote_columns_n=[x+'_n' for x in vote_columns]
train_data[vote_columns_n].value_counts()


#Defining a function to classify eeg data according to vote distribution

def classify_pattern(x):
    named_columns=['seizure_vote_n',  'lpd_vote_n',  'gpd_vote_n',  'lrda_vote_n',  'grda_vote_n']
    #Agreeing on one type
    if np.any(x[vote_columns_n]>0.95):
        pattern='idealized'
    #Agreeing on other and named
    elif np.any(x[named_columns]>0.45) and x['other_vote_n']>0.45:
        pattern='proto'
    else:
        pattern= 'edge'
    return pattern


train_data['pattern']=train_data.apply(lambda x:classify_pattern(x),axis=1)
train_data



#Visualizing distribution of pattern
train_data['pattern'].value_counts().plot(kind='bar')


#Checking the test data
test_data=pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
test_data


print(f'Nos. of train eeg files:{len(os.listdir(train_eegs_dir))}')
eegs_list=os.listdir(train_eegs_dir)


sample_eeg=train_eegs_dir/eegs_list[2]
sample_eeg


eeg_id=re.search(r'[0-9]+',str(sample_eeg)).group()
train_data[train_data['eeg_id']==int(eeg_id)]


#Reading a sample eeg file

pd.read_parquet(sample_eeg)



#Defining a function to plot EEG

def plot_eeg(eeg_path):
    sample_eeg=pd.read_parquet(eeg_path)

    for i,col in enumerate(sample_eeg.columns):
        fig=plt.figure(figsize=(5,30))
        ax=plt.subplot(20,1,i+1)
        ax.plot(sample_eeg[col])
        ax.set_xlabel(col)
        plt.grid()


plot_eeg(eeg_path=sample_eeg)


#Checking the test eeg
print(f'Nos. of test eeg files:{len(os.listdir(test_eegs_dir))}')
test_eegs_list=os.listdir(test_eegs_dir)


print(f'Nos. of spectrogram files:{len(os.listdir(train_spectrogram_dir))}')
spg_list=os.listdir(train_spectrogram_dir)


#Reading a sample spectrogram file
sample_spectrogram=train_spectrogram_dir/spg_list[2]
pd.read_parquet(sample_spectrogram)


def plot_spectrogram(spectrogram_path):
    sample_spect = pd.read_parquet(spectrogram_path)
    
    split_spect = {
        "LL": sample_spect.filter(regex='^LL', axis=1),
        "RL": sample_spect.filter(regex='^RL', axis=1),
        "RP": sample_spect.filter(regex='^RP', axis=1),
        "LP": sample_spect.filter(regex='^LP', axis=1),
    }
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()
    label_interval = 5
    for i, split_name in enumerate(split_spect.keys()):
        ax = axes[i]
        img = ax.imshow(np.log(split_spect[split_name]).T, cmap='viridis', aspect='auto', origin='lower')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Log(Value)')
        ax.set_title(split_name)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")

        ax.set_yticks(np.arange(len(split_spect[split_name].columns)))
        ax.set_yticklabels([column_name[3:] for column_name in split_spect[split_name].columns])
        frequencies = [column_name[3:] for column_name in split_spect[split_name].columns]
        ax.set_yticks(np.arange(0, len(split_spect[split_name].columns), label_interval))
        ax.set_yticklabels(frequencies[::label_interval])
    plt.tight_layout()
    plt.show()



plot_spectrogram(sample_spectrogram)


#Getting the first eeg_id and spectogram_id
sample_eeg_id=train_data['eeg_id'][0]
sample_spectogram_id=train_data['spectrogram_id'][0]

sample_data=train_data[train_data['eeg_id']==sample_eeg_id]
sample_data


#Reading eeg data
sample_eeg=train_eegs_dir/(str(sample_eeg_id)+'.parquet')
pd.read_parquet(sample_eeg)



#Visualizing eeg data
plot_eeg(sample_eeg)


#Reading spectogram data
sample_spg=train_spectrogram_dir/(str(sample_spectogram_id)+'.parquet')
pd.read_parquet(sample_spg)


plot_spectrogram(sample_spg)


#Checking the occurence of spectrograms against eeg
train_data.groupby(['eeg_id'])['spectrogram_id'].nunique().unique()


train_data.groupby(['spectrogram_id'])['eeg_id'].nunique().unique()


#Each spectrogram ID may have multiple expert_consensus
train_data.groupby(['spectrogram_id'])['expert_consensus'].nunique().value_counts()


#Each eeg ID may have multiple expert_consensus
train_data.groupby(['eeg_id'])['expert_consensus'].nunique().value_counts()


train_data


sample_eeg_id=train_data['eeg_id'][0]
sample_eeg=train_eegs_dir/(str(sample_eeg_id)+'.parquet')
sample_df=pd.read_parquet(sample_eeg)
sample_df


sample_train_data=train_data[train_data['eeg_id']==sample_eeg_id]
sample_train_data


import torch
from torch.utils.data import Dataset,DataLoader,Subset


device='cuda' if torch.cuda.is_available() else 'cpu'


class EEGDataset(Dataset):
    
    def __init__(self,data,eeg_dir_path):
        self.data=data
        self.eeg_dir_path=eeg_dir_path
        self.targets=['seizure_vote',	'lpd_vote',	'gpd_vote',	'lrda_vote',	'grda_vote',	'other_vote']
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        sample_eeg_id=self.data.iloc[index]['eeg_id'] #Getting the id
        sample_eeg_offset=self.data.iloc[index]['eeg_label_offset_seconds']  #Getting the offset

        #Getting the  targets
        sample_eeg_target=self.data.iloc[index][self.targets]

        sample_eeg=self.eeg_dir_path/(str(sample_eeg_id)+'.parquet')  #Getting the EEG file 
        sample_eeg_df=pd.read_parquet(sample_eeg)  #Reading the parquet

        #Imputing values
        
        #sample_eeg_df=sample_eeg_df.bfill(axis=0) #Filling null values across rows
        #sample_eeg_df=sample_eeg_df.ffill(axis=0)

        #Getting the EEG data
        X=self.get_features(eeg_df=sample_eeg_df,eeg_offset=sample_eeg_offset).T #Transposing to get in the shape of (Channel,Timesteps)
        
        y=np.array(sample_eeg_target.values.astype('float32')) 
        y=torch.tensor(y)
        y=y.softmax(dim=-1)  #Since we want probability distribution
        

        return torch.tensor(X),y


    def get_features(self,eeg_df,eeg_offset):

        #Each row in eeg_df contains 1 of 200 samples per second
        
        #Checking for existence of atleast 50 secs or 10000 rows of data
        #if int(eeg_offset)*200+50*200<=len(eeg_df):
        truncated_df=eeg_df.iloc[int(eeg_offset)*200:int(eeg_offset)*200+50*200] #Collecting 50 secs of data
        eeg_array=truncated_df.to_numpy()  #Shape:(timesteps,features)
        
            
        #else:/
            #truncated_df=eeg_df.iloc[int(eeg_offset)*200:] #Collecting till last
            #eeg_array=truncated_df.to_numpy()

            #Padding to 10000 time steps
            #rows_to_add = 10000 - eeg_array.shape[0]
            #padding = ((0, rows_to_add), (0, 0))  # Pad rows at the end, no padding for columns
            #eeg_array = np.pad(eeg_array, padding, mode='mean')
        return eeg_array

    

    def target_dict(self):
        return {i:t for (i,t) in enumerate(self.targets)}
        
            
            
            
        

        
 


#Defining a function to get null indexes
def get_null_indexes(data,eeg_dir_path):
    null_index_list=[]
    for index in range(len(data)):
        eeg_id=data.iloc[index]['eeg_id'] #Getting the id
        eeg_offset=data.iloc[index]['eeg_label_offset_seconds']  #Getting the offset

        eeg_file=eeg_dir_path/(str(eeg_id)+'.parquet')  #Getting the EEG file 
        eeg_df=pd.read_parquet(eeg_file)  #Reading the parquet


        truncated_df=eeg_df.iloc[int(eeg_offset)*200:int(eeg_offset)*200+50*200] #Collecting 50 secs of data
        eeg_array=truncated_df.to_numpy()  #Shape:(timesteps,features).T #Transposing to get in the shape of (Channel,Timesteps)

        if np.isnan(eeg_array).any():
            null_index_list.append(index)

    return null_index_list
                


#Getting null indices

#null_index_list=get_null_indexes(data=train_data,eeg_dir_path=train_eegs_dir)


#Null list location
null_index_dir=Path('/kaggle/working')
null_index_path=null_index_dir/'null_index.json'


#Null index download path
null_index_download_path=Path('/kaggle/input')/'null-index-json/null_index.json'


#Storing files in an object
'''
with open(null_index_path,'w') as f:
    json.dump(null_index_list,f)
'''


#Loading the list
with open(null_index_download_path,'r') as f:
    null_list=json.load(f)
    


print(f'No of null index:{len(null_list)}')


#Getting the non-null indices
non_null_indices=list(set(range(len(train_data)))-set(null_list))
print(f'All indices:{len(train_data)}')
print(f'Non null indices:{len(non_null_indices)}')


#Splitting the indices
train_split=0.8

train_indices=np.random.choice(non_null_indices,int((len(non_null_indices)*train_split)),replace=False).tolist()
val_indices=list(set(non_null_indices)-set(train_indices))




#Splitting the data based on indices

train_set=train_data.iloc[train_indices]
val_set=train_data.iloc[val_indices]


#Initialializing the dataset
train_eeg_ds=EEGDataset(data=train_set,eeg_dir_path=train_eegs_dir)
val_eeg_ds=EEGDataset(data=val_set,eeg_dir_path=train_eegs_dir)


#Checking length of dataset
print(f'Length of train dataset:{len(train_eeg_ds)}')
print(f'Length of val dataset:{len(val_eeg_ds)}')


#Checking shape of dataset
print(f'Shape of X in dataset:{train_eeg_ds[0][0].shape}')
print(f'Shape of y in dataset:{train_eeg_ds[0][1].shape}')


#Creating a subdataset for research purpose

train_sub_size=30000
val_sub_size=5000

train_sub_indices=np.random.choice(range(len(train_eeg_ds)),train_sub_size,replace=False)
val_sub_indices=np.random.choice(range(len(val_eeg_ds)),val_sub_size,replace=False)

train_sub_ds=Subset(train_eeg_ds,indices=train_sub_indices)
val_sub_ds=Subset(val_eeg_ds,indices=val_sub_indices)
print(f'Length of sub dataset:{len(train_sub_ds)}')


batch_size=32
#Creating a sub dataloader
train_sub_dl=DataLoader(train_sub_ds,batch_size=batch_size,shuffle=True)
val_sub_dl=DataLoader(val_sub_ds,batch_size=batch_size,shuffle=False)


print(f'Shape of X in dataloader:{next(iter(train_sub_dl))[0].shape}')
print(f'Shape of y in dataloader:{next(iter(train_sub_dl))[1].shape}')



sample_dl_X=next(iter(train_sub_dl))[0].to(device)
print(f'Shape of sample_dl={sample_dl_X.shape}')



#Getting full validation dataloader
val_eeg_dl=DataLoader(val_eeg_ds,batch_size=64)


#Getting the dict

target_dict=val_eeg_ds.target_dict()


import torch
from torch import nn
from torch.nn import functional as F

import torchinfo


import gc

gc.collect()

torch.cuda.empty_cache()


def train_model(train_dl,val_dl,model,loss_fn,optimizer,epochs,model_path=None,scheduler=None,best_loss=None):
    epoch_train_loss=[]
    epoch_val_loss=[]
    if best_loss:
        best_loss=best_loss
    else:
        best_loss=np.inf
    
    for i in range(epochs):
        #Training the model
        model.train()
        train_loss=0
        for n,(X,y) in enumerate(train_dl):
            X,y=X.to(device),y.to(device)
            y_pred=model(X)  #y_pred is softmax
            
            loss=loss_fn(torch.log(y_pred),y)  
    
            train_loss+=loss*y.shape[0]
            #Zero grad the optimizer
            optimizer.zero_grad()
            #Backpropagate
            loss.backward()
            #Updating parameters
            optimizer.step()
            #Tuning the scheduler
            if scheduler:
                scheduler.step()
            if n%10==0:
                print(f'Batch:{i}_{n} | Batch loss:{loss.item():.2f}')

           

        train_loss=(train_loss/len(train_dl.dataset)).item()
        print(f'Epoch:{i} | Train loss:{train_loss:.2f}')
        epoch_train_loss.append(train_loss)

        #Evaluating model
        model.eval()
        val_loss=0
        with torch.no_grad():
            for X,y in val_dl:
                X,y=X.to(device),y.to(device)
                y_pred=model(X)  #y_pred is softmax
                loss=loss_fn(torch.log(y_pred),y)
                val_loss+=loss*y.shape[0]
            val_loss=(val_loss/len(val_dl.dataset)).item()
            
            print(f'Epoch:{i} | Val loss:{val_loss:.2f}')
            epoch_val_loss.append(val_loss)
        #Saving model
        if model_path:
            model_path.parent.mkdir(parents=True,exist_ok=True)
            if val_loss<best_loss:
                torch.save(model.state_dict(),model_path)  #saving the state dict
                best_loss=val_loss
                
                
            
    return epoch_train_loss,epoch_val_loss




#Defining evaluation function

def eval_model(model,val_dl,loss_fn):
    model.eval()
    val_loss=0
    with torch.no_grad():
        for (X,y) in val_dl:
            X,y=X.to(device),y.to(device)
            y_pred=model(X)
            loss=loss_fn(torch.log(y_pred),y)
            val_loss+=loss*y.shape[0]
        val_loss=(val_loss/len(val_dl.dataset)).item()
    return val_loss



#Defining a function to  get prediction
def predict_labels(model,eeg_data,target_dict):
    model.eval()
    pred=model(eeg_data).detach().cpu().numpy()
    return pd.DataFrame(pred,columns=target_dict.values())



#Defining a function to  get load pre-trained models
def load_pretrained(model,state_dict_path):
    if torch.cuda.is_available()==False:
        map_location=torch.device('cpu')
    else:
        map_location=torch.device('cuda')
    
    model.load_state_dict(torch.load(state_dict_path,weights_only=True,map_location=map_location))
    return model


#Paths for pretrained model
eeg_model_dir=Path('/kaggle/input')
eeg_simple_state=eeg_model_dir/'eeg-simple-model'/'pytorch'/'default'/'3'/'eeg_model.pth'
eeg_resnet_state=eeg_model_dir/'eeg-resnet-model'/'pytorch'/'default'/'2'/'eeg_r_model.pth'


class EEGNet(nn.Module):

    def __init__(self,num_features,num_targets,num_temp_filters=40,num_spatial_filters=40,dropout_rate_cn=0.1,dropout_rate_fc=0.3):
        super().__init__()
        self.dropout_rate_cn=dropout_rate_cn
        self.dropout_rate_fc=dropout_rate_fc
        self.num_targets=num_targets
        #Normalizing across channels
        self.bn1=nn.BatchNorm1d(num_features=num_features) 
        
        #Using a filter to convolve across temporal direction
        self.temp_conv=nn.Conv2d(in_channels=1,out_channels=num_temp_filters,kernel_size=(1,64),bias=False)
        #Normalizing across temporal features
        self.bn2=nn.BatchNorm2d(num_features=num_temp_filters)
        
        #Using a filter to convolve across spatial direction
        self.depth_conv=nn.Conv2d(in_channels=num_temp_filters,out_channels=num_spatial_filters,kernel_size=(num_features,1),bias=False)
        #Normalizing across spatial features
        self.bn3=nn.BatchNorm2d(num_features=num_spatial_filters)
        #Average pooling
        self.avgpool1=nn.AvgPool2d(kernel_size=(1,8))
        
        
        
        #Using a filter to convolve across features
        self.sep_conv=nn.Conv2d(in_channels=num_spatial_filters,out_channels=num_spatial_filters,kernel_size=(1,16),bias=False)
        #Normalizing across  features
        self.bn4=nn.BatchNorm2d(num_features=num_spatial_filters) 
        #Average pooling
        self.avgpool2=nn.AvgPool2d(kernel_size=(1,16))
        
        #Applying LSTM
        self.lstm=None

        #Normalizing across  features
        self.bn5=None 

        
        #Linear layer 1
        self.linear1=None
        #Applying batch norm
        self.bn6=None
        #Linear layer 2
        self.linear2=None
        #Applying batch norm
        self.bn7=None

        #Dropout
        self.dropout_cn=nn.Dropout(p=dropout_rate_cn)
        self.dropout_fc=nn.Dropout(p=dropout_rate_fc)
    
    
    

    def forward(self,x):
        x=self.bn1(x)     #Input shape (batch_size,features,timesteps)
        x=torch.unsqueeze(x,1)  #Output shape (batch_size,1,features,timesteps)
        
        x=self.temp_conv(x)     #Output shape (batch_size,filters,features,timesteps)
        x=self.bn2(x)           
        x=F.elu(x)
        
        x=self.depth_conv(x)    #Output shape (batch_size,features,1,timesteps)
        x=self.bn3(x)
        x=F.elu(x)
        x=self.avgpool1(x)
        
       
        
        x=self.sep_conv(x)     #Output shape (batch_size,features,1,timesteps)
        x=self.bn4(x)
        x=F.elu(x)
        x=self.avgpool2(x)
       
        
       
        x=x.squeeze(-2)        #Output shape (batch_size,features,timesteps)
        
        lstm_in=x.permute(0,2,1)     #Output shape (batch_size,timesteps,features)

        lstm_in=self.dropout_cn(lstm_in)
        
        

        if self.lstm==None:
            features=lstm_in.shape[-1]
            self.lstm=nn.LSTM(input_size=features, hidden_size=features, num_layers=1, bias=True, 
                              batch_first=True, dropout=0.0, bidirectional=False).to(device)

        
        lstm_out=self.lstm(lstm_in)[0][:,-1,:]   #Output shape (batch_size,features) 
        
        
        if self.bn5==None:
            in_features = lstm_out.shape[1]
            self.bn5 = nn.BatchNorm1d(num_features=in_features).to(device)
        lstm_out=self.bn5(lstm_out)

       
        x=nn.Flatten()(x)  #Output shape (batch_size,features)
        #Concatenating lsmtm output with original features
        x=torch.cat([x,lstm_out],dim=-1)
        x=self.dropout_fc(x)
       
        if self.linear1==None:
            in_features = x.shape[1]
            self.linear1 = nn.Linear(in_features,in_features//10 ).to(device)
        x=self.linear1(x)
        if self.bn6==None:
            in_features = x.shape[1]
            self.bn6 = nn.BatchNorm1d(num_features=in_features).to(device)
        x=self.bn6(x)
        x=F.elu(x)
        x=self.dropout_fc(x)
            
        if self.linear2==None:
            in_features = x.shape[1]
            self.linear2 = nn.Linear(in_features,self.num_targets ).to(device)
        x=self.linear2(x)


        
        x=F.softmax(x,dim=-1)      
        
        return x
        



#Initililizing the model
eegnet=EEGNet(num_features=20,num_targets=6,num_temp_filters=20,num_spatial_filters=40,dropout_rate_cn=0.1,dropout_rate_fc=0.3).to(device)


out=eegnet(sample_dl_X)
print(f'Shape of output:{out.shape}')



torchinfo.summary(eegnet,input_data=sample_dl_X)


#Loading pre-trained model
eegnet=load_pretrained(model=eegnet,state_dict_path=eeg_simple_state)


#Defining the hyperparameters
config_eegnet={
    'train_dl':train_sub_dl,
    'val_dl':val_sub_dl,
    'model':eegnet,
    'model_path':Path('/kaggle/working')/'model_dir'/'eeg_model.pth',
    'epochs':5,
    'best_loss':np.inf
}




#Defining loss and optimization functions
loss_fn=nn.KLDivLoss(reduction="batchmean")

optimizer=torch.optim.AdamW(eegnet.parameters(), lr=0.001)

scheduler=torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=config_eegnet['epochs'],
        steps_per_epoch=len(config_eegnet['train_dl']),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=1000,
    )


train_model(train_dl=config_eegnet['train_dl'],
            val_dl=config_eegnet['val_dl'],
            model=eegnet,loss_fn=loss_fn,optimizer=optimizer,
            epochs=config_eegnet['epochs'],
            scheduler=scheduler,
            model_path=config_eegnet['model_path'],
            best_loss=config_eegnet['best_loss'])


#Evaluating model
eval_model(model=eegnet,val_dl=val_eeg_dl,loss_fn=loss_fn)


class ResNet_1D_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,dropout, downsampling):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding='same', bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out




class EEGNet_resnet(nn.Module):
    def __init__(self, num_features, num_targets, temporal_filter_lengths, num_temp_filters=24, dropout_rate_cn=0.1,dropout_rate_fc=0.3,res_blocks=6):
        super().__init__()
        self.num_temp_filters=num_temp_filters
        
        self.dropout_rate_cn = dropout_rate_cn
        self.dropout_rate_fc = dropout_rate_fc
        self.num_targets = num_targets
        self.temporal_filter_lengths = temporal_filter_lengths  # Store filter lengths
        self.res_blocks=res_blocks

        # Normalizing across channels
        self.bn1 = nn.BatchNorm1d(num_features=num_features)

        # Parallel convolutions with multiple filter lengths
        self.parallel_convs = nn.ModuleList()
        for filter_length in temporal_filter_lengths:
            self.parallel_convs.append(
                nn.Sequential(
                    #Using a filter to convolve across temporal direction
                    nn.Conv1d(in_channels=num_features, out_channels=num_temp_filters, kernel_size=filter_length,padding='same',bias=False),
                    nn.BatchNorm1d(num_features=num_temp_filters),
                    nn.ReLU(),
                    nn.AvgPool1d(kernel_size=4, stride=2, padding=0)
                     )
                    )
       
        # Using a filter to convolve across features
        self.sep_conv = nn.Conv1d(in_channels=num_temp_filters*len(temporal_filter_lengths), out_channels=num_temp_filters*len(temporal_filter_lengths), kernel_size=8,stride=2,bias=False)
        # Normalizing across  features
        self.bn4 = nn.BatchNorm1d(num_features=num_temp_filters*len(temporal_filter_lengths))
        # Average pooling
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)

        #Making resnet block
        self.block = self._make_resnet_layer(kernel_size=16, stride=1,blocks=res_blocks, dropout=dropout_rate_cn)

        # Applying LSTM
        self.lstm = None

        # Normalizing across  features
        self.bn5 = None

        # Linear layer 1
        self.linear1 = None
        # Applying batch norm
        self.bn6 = None
        # Linear layer 2
        self.linear2 = None
        # Applying batch norm
        self.bn7 = None
        # Linear layer 3
        self.linear3 = None
        # Applying batch norm
        self.bn8= None

        #Dropout
        self.dropout_cn=nn.Dropout(p=dropout_rate_cn)
        self.dropout_fc=nn.Dropout(p=dropout_rate_fc)

        #ReLU
        self.relu=nn.ReLU()
    
    #Function to make resnet layers
    def _make_resnet_layer(self, kernel_size, stride, blocks,dropout):
        planes=self.num_temp_filters*len(self.temporal_filter_lengths)
        layers = []
        downsample = None
       

        for i in range(blocks):
            downsampling = nn.Sequential(
                    nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
                )
            layers.append(ResNet_1D_Block(in_channels=planes, out_channels=planes, kernel_size=kernel_size,
                                       stride=stride, dropout=dropout, downsampling=downsampling))

        return nn.Sequential(*layers)
        

    def forward(self, x):
        x = self.bn1(x)  # Input shape (batch_size, features, timesteps)
        
        # Apply each temporal convolution in parallel and concatenate the results
        temp_outputs = []
        for parallel_conv in self.parallel_convs:
            temp_output=parallel_conv(x)
            temp_outputs.append(temp_output)
          
        x = torch.cat(temp_outputs, dim=1)  # Output shape (batch_size,features*temporal_length, timesteps)

        
        x = self.sep_conv(x)  # Output shape (batch_size, features, timesteps)
        x = self.bn4(x)
        
        #Apply resnet convolution blocks
        x = self.block(x)  # Output shape (batch_size, features, timesteps)
        
        if self.bn5 is None:
            in_features = x.shape[1]
            self.bn5 = nn.BatchNorm1d(num_features=in_features).to(device)
        x = self.bn5(x)  # Output shape (batch_size, features, timesteps)
        
        #Apply LSTM 
        lstm_in = x.permute(0, 2, 1)  # Output shape (batch_size, timesteps, features)
         
        if self.lstm is None:
            lstm_features = lstm_in.shape[-1]
            self.lstm = nn.LSTM(input_size=lstm_features, hidden_size=lstm_features , num_layers=1, bias=True,
                                        batch_first=True, bidirectional=False).to(device)

        #taking last timestep
        lstm_out = self.lstm(lstm_in)[0][:,-1,:]  # Output shape (batch_size, features)

        if self.bn6 is None:
            in_features = lstm_out.shape[1]
            self.bn6 = nn.BatchNorm1d(num_features=in_features).to(device)
        lstm_out = self.bn6(lstm_out)

        #Flattening resnet output
        x = nn.Flatten()(x)  # Output shape (batch_size, features)
        
        #Concatenating resnet  and lstm output
        x=torch.cat([x,lstm_out],dim=-1)    
        x = self.dropout_fc(x)
       
        #Applying linear layer
        if self.linear1 is None:
            in_features = x.shape[1]
            self.linear1 = nn.Linear(in_features,in_features//8 ).to(device)
        x = self.linear1(x)
        if self.bn7 is None:
            in_features = x.shape[1]
            self.bn7 = nn.BatchNorm1d(num_features=in_features).to(device)
        x = self.bn7(x)
        x = self.relu(x)
        
       
        x = self.dropout_fc(x)

        if self.linear2 is None:
            in_features = x.shape[1]
            self.linear2 = nn.Linear(in_features, self.num_targets).to(device)
        x = self.linear2(x)
     
      
        x = F.softmax(x, dim=-1)
       
        return x



#Initililizing the model
eegnet_r=EEGNet_resnet(num_features=20, num_targets=6, temporal_filter_lengths=[4,8,12], num_temp_filters=24, dropout_rate_cn=0.1,dropout_rate_fc=0.3,res_blocks=7).to(device)


out=eegnet_r(sample_dl_X)
print(f'Shape of output:{out.shape}')



torchinfo.summary(eegnet_r,input_data=sample_dl_X)


#Loading pre-trained model
eegnet_r=load_pretrained(model=eegnet_r,state_dict_path=eeg_resnet_state)


#Defining the hyperparameters
config_eegnet_r={
    'train_dl':train_sub_dl,
    'val_dl':val_sub_dl,
    'model':eegnet_r,
    'model_path':Path('/kaggle/working')/'model_dir'/'eeg_r_model.pth',
    'epochs':3,
    'best_loss':0.78 #The best loss for pre-trained parameter
}


#Defining loss and optimization functions
loss_fn=nn.KLDivLoss(reduction="batchmean")

optimizer=torch.optim.AdamW(eegnet_r.parameters(), lr=0.5*1e-4)

scheduler=torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.5*1e-4,
        epochs=config_eegnet_r['epochs'],
        steps_per_epoch=len(config_eegnet_r['train_dl']),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )


train_model(train_dl=config_eegnet_r['train_dl'],
            val_dl=config_eegnet_r['val_dl'],
            model=eegnet_r,
            loss_fn=loss_fn,optimizer=optimizer,
            epochs=config_eegnet_r['epochs'],
            scheduler=scheduler,
            model_path=config_eegnet_r['model_path'],
            best_loss=config_eegnet_r['best_loss'])


#Evaluating model
eval_model(model=eegnet_r,val_dl=val_eeg_dl,loss_fn=loss_fn)


#Loading a sample data
sample_index=np.random.choice(np.arange(len(val_eeg_ds)),1).item()

sample_data_X=val_eeg_ds[sample_index][0].unsqueeze(0).to(device)
sample_data_y=val_eeg_ds[sample_index][1]


#Actual
pd.DataFrame(sample_data_y.numpy().reshape(1,6),columns=target_dict.values())


#Predicting with eegnet resnet model
predict_labels(model=eegnet_r,eeg_data=sample_data_X,target_dict=target_dict)


train_data


sample_spg_id=train_data['spectrogram_id'][0]
sample_spg=train_spectrogram_dir/(str(sample_spg_id)+'.parquet')
pd.read_parquet(sample_spg)


train_data[train_data['spectrogram_id']==sample_spg_id]





class SPGDataset(Dataset):
    
    def __init__(self,data,eeg_dir_path):
        self.data=data
        self.spg_dir_path=spg_dir_path
        self.targets=['seizure_vote',	'lpd_vote',	'gpd_vote',	'lrda_vote',	'grda_vote',	'other_vote']
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        sample_spg_id=self.data.iloc[index]['spectrogram_id'] #Getting the id
        sample_spg_offset=self.data.iloc[index]['spectrogram_label_offset_seconds']  #Getting the offset

        #Getting the  targets
        sample_spg_target=self.data.iloc[index][self.targets]

        sample_spg=self.spg_dir_path/(str(sample_spg_id)+'.parquet')  #Getting the SPG file 
        sample_spg_df=pd.read_parquet(sample_spg)  #Reading the parquet
        l
        sample_spg_df=sample_eeg_df.bfill(axis=0) #Filling null values across rows
        sample_spg_df=sample_eeg_df.ffill(axis=0)

        #Getting the SPG data
        X=self.get_features(spg_df=sample_spg_df,spg_offset=sample_spg_offset).T #Transposing to get in the shape of (Channel,Timesteps)
        
        y=np.array(sample_spg_target.values.astype('float32')) 
        y=torch.tensor(y)
        y=y.softmax(dim=-1)  #Since we want probability distribution
        

        return torch.tensor(X),y


    def get_features(self,spg_df,spg_offset):

        #Each row in spg_df contains 2 sec of data and each upto 10 min or 600 sec of data
        
        #Checking for existence of atleast 50 secs or 10000 rows of data
        #if int(eeg_offset)*200+50*200<=len(eeg_df):
        
        truncated_df=spg_df.iloc[int(spg_offset)/2:(int(spg_offset)+600)/2] #Collecting 600 secs of data
        
        eeg_array=truncated_df.to_numpy()  #Shape:(timesteps,features)
        
            
        #else:
            #truncated_df=eeg_df.iloc[int(eeg_offset)*200:] #Collecting till last
            #eeg_array=truncated_df.to_numpy()

            #Padding to 10000 time steps
            #rows_to_add = 10000 - eeg_array.shape[0]
            #padding = ((0, rows_to_add), (0, 0))  # Pad rows at the end, no padding for columns
            #eeg_array = np.pad(eeg_array, padding, mode='mean')
        return eeg_array

    

    def target_dict(self):
        return {i:t for (i,t) in enumerate(self.targets)}
        
            
            
            
        


def plot_spectrogram(spectrogram_path):
    sample_spect = pd.read_parquet(spectrogram_path)
    
    split_spect = {
        "LL": sample_spect.filter(regex='^LL', axis=1),
        "RL": sample_spect.filter(regex='^RL', axis=1),
        "RP": sample_spect.filter(regex='^RP', axis=1),
        "LP": sample_spect.filter(regex='^LP', axis=1),
    }
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()
    label_interval = 5
    for i, split_name in enumerate(split_spect.keys()):
        ax = axes[i]
        img = ax.imshow(np.log(split_spect[split_name]).T, cmap='viridis', aspect='auto', origin='lower')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Log(Value)')
        ax.set_title(split_name)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")

        ax.set_yticks(np.arange(len(split_spect[split_name].columns)))
        ax.set_yticklabels([column_name[3:] for column_name in split_spect[split_name].columns])
        frequencies = [column_name[3:] for column_name in split_spect[split_name].columns]
        ax.set_yticks(np.arange(0, len(split_spect[split_name].columns), label_interval))
        ax.set_yticklabels(frequencies[::label_interval])
    plt.tight_layout()
    plt.show()





