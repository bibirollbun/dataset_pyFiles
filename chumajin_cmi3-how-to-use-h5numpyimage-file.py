import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import polars as pl
import h5py
from torch.utils.data import DataLoader, TensorDataset,Dataset
import matplotlib.pyplot as plt
import seaborn as sns


train = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train.head(3)


images = h5py.File("/kaggle/input/cmi3-2d-tof-numpy-images/2dimages.h5","r")


rowid = "SEQ_000007_000001"


images[rowid][:]


for a in range(5):
    plt.figure(figsize=(7,5))
    sns.heatmap(images[rowid][:][a])


images.close()


class PytorchDataSet(Dataset):
    
    def __init__(self,df,h5path):

        self.df = df

        self.rowid = self.df["row_id"]
        self.target = self.df["gesture"]

        self.images = h5py.File(h5path,"r")


    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self,idx):

        rowid = self.rowid[idx]
        image = self.images[rowid][:]
        
        return {"image":image,"target":self.target[idx]}


train_dataset = PytorchDataSet(train,'/kaggle/input/cmi3-2d-tof-numpy-images/2dimages.h5')


train_dataset[0]


for a in range(5):
    plt.figure(figsize=(7,5))
    sns.heatmap(train_dataset[500]["image"][a])




