import pandas as pd
import numpy as np
import librosa
import glob
import pandas.api.types
import torch 
import torch.nn as nn
import os
import random
from matplotlib import pyplot as plt
import seaborn as sns
import timm
import kaggle_metric_utilities
import sklearn.metrics

from warnings import filterwarnings
filterwarnings("ignore")
from ast import literal_eval 


class Config:
    train_dir="/kaggle/input/birdclef-2025/train_audio"
    train_csv="/kaggle/input/birdclef-2025/train.csv"
    sample_csv="/kaggle/input/birdclef-2025/sample_submission.csv"
    test_soundscapes="/kaggle/input/birdclef-2025/test_soundscapes"
    seed = 42
    sr = int(32e3)
    num_classes= 206
    n_fft= 1024
    hop_length=500
    n_mels=128
    fmin= 50
    fmax=16000
    power=2
    


def set_seed(seed: int =Config.seed)-> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Set seed: {seed}")

set_seed()

    



data_df = pd.read_csv(Config.train_csv)
for col in ('secondary_labels','type'):
    data_df[col]=data_df[col].apply(lambda x:"###".join(literal_eval(x)))
data_df['filename']= data_df['filename'].apply(lambda x: Config.train_dir + "/"+x)
data_df.sample(10)


data_df.isnull().sum()


plt.figure(figsize=(20,5))
sns.histplot(data_df, x='rating')
plt.xticks(np.arange(0,5.5,0.5))
plt.show();


for r in range (0,7):
    plt.figure(figsize=(20,5))
    sns.histplot(data_df[data_df['rating']==float(r)], x='primary_label')
    plt.title(f"rating {r}")
    plt.xticks(rotation=90)
    plt.show();


durations=[]
for idx, row in data_df.sample(100).iterrows():
    data, _ = librosa.load(row['filename'], sr = Config.sr)
    durations.append(librosa.get_duration(y=data, sr = Config.sr))
d_df = pd.DataFrame(columns=["durations"], data=durations)
plt.figure(figsize=(12,5))
plt.title("dist of audio lenghts")
sns.histplot(d_df,x="durations")
plt.show();
d_df.describe()


def show_signal(file_path):
    class_,collector =file_path.split("/")[-2:]
    y,sr=librosa.load(file_path,sr=Config.sr)
    fig,axes=plt.subplots(2,2,figsize=(20,10))
    fig.suptitle(f"Class:{class_}|Collector:{collector}",fontsize=16)ch
    librosa.display.waveshow(y,sr=sr,ax=axes[0,0])
    axes[0,0].set_title("raw signal")
#short time fourier transform 
    ft=np.abs(librosa.stft(
        y,
        n_fft=Config.n_fft,
        hop_length=Config.hop_length
        
    ))
    im1= librosa.display.specshow(
        ft,
        sr=sr,
        x_axis='time',
        y_axis='linear',
        ax=axes[0,1]
    )
    fig.colorbar(im1,ax=axes[0,1])
    axes[0,1].set_title("spectogram")

    ft_db=librosa.amplitude_to_db(ft,ref=np.max)
    im2=librosa.display.specshow(
        ft_db,
        sr=sr,
        x_axis='time',
        y_axis='log',
        ax=axes[1,0]
    )
    fig.colorbar(im1,ax=axes[1,0])
    axes[1,0].set_title("Log scaled")

    mel_sp=librosa.feature.melspectrogram(
        y=y,
        sr=Config.sr,
        fmin=Config.fmin,
        fmax=Config.fmax,
        power=Config.power,
        n_mels=Config.n_mels,  
    )
    mel_sp=librosa.power_to_db(mel_sp,ref=np.max)
    im3=librosa.display.specshow(
        mel_sp,
        y_axis='mel',
        sr= Config.sr,
        fmin=Config.fmin,
        x_axis='time',
        fmax=Config.fmax,
        ax=axes[1,1]
    )
    fig.colorbar(im1,ax=axes[1,1])
    axes[1,1].set_title("mel spectogram")
    
    plt.show()
show_signal(data_df['filename'].values[0])


for idx, row in data_df.sample(5).iterrows(): show_signal(row['filename'])


label_mapper={
    label:idx
    for idx,label in enumerate (sorted(data_df['primary_label'].unique()))
}
rev_mapper={
    idx:label
    for label,idx in label_mapper.items()
}

class BirdClefDataset(torch.utils.data.Dataset):
    def __init__(self,df,mode="train"):
        self.df=df
        self.mode=mode
        
    def __len__(self):return len(self.df)

    
    def process(self, audio_path):
        data,_=librosa.load(audio_path, sr= Config.sr)
        data=data*1024
        chunk_duration=10
        min_len= chunk_duration* Config.sr
        # less than min 
        if len(data)< min_len:
            cnt=int(np.ceil(min_len/len(data)))
            data=np.tile(data,cnt)
            
        #len div by the min
        leftover=len(data)%min_len
        if leftover>0:
            front_crop=leftover//2
            back_crop=leftover-front_crop
            data=data[front_crop:len(data)-back_crop]

        data=data[:min_len]
        data=data.reshape(-1,min_len)
        mel_sp=librosa.feature.melspectrogram(
            y=data,
            sr=Config.sr,
            fmin=Config.fmin,
            fmax=Config.fmax,
            power=Config.power,
            n_mels=Config.n_mels,
            n_fft=Config.n_fft,
            hop_length=Config.hop_length
        )
        mel_sp=librosa.power_to_db(mel_sp,ref=1)


        #normalize the feature values

        eps=1e-12
        mel_sp=(mel_sp-mel_sp.min())/(mel_sp.max()-mel_sp.min()+eps)
        mel_sp=mel_sp[:,:,:640]
        return mel_sp
        
    def __getitem__ (self,idx):
        row=self.df.loc[idx,:]
        filename= row['filename']
        x=self.process(filename)

        if self.mode=="train":
            y= label_mapper[row['primary_label']]
            return x,y
        return x

            


mel_sp=BirdClefDataset(data_df).process(data_df['filename'].values[0])
print(mel_sp.shape)
#transf from c,h,w->h,w,c
plt.imshow(mel_sp.reshape(128,640,-1))
plt.show();


class Model(nn.Module):
    def __init__(self,model_name:str):
        super().__init__()
        self.base_model=timm.create_model(
            model_name=model_name,
            num_classes=Config.num_classes,
            pretrained= False,
            in_chans=1
        )
    def forward(self,x):
        return self.base_model(x)
    




