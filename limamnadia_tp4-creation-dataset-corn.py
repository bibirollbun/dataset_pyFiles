import os
import pandas as pd
import shutil
import tqdm


train=pd.read_csv("/kaggle/input/kaggle-pog-series-s01e03/corn/train.csv")
train


train["label"]


train["label"].tolist()


train.loc[0,"image"]


train["label"].unique()


train["label"]=="pure"


train[train["label"]=="pure"]


train[train["label"]=="pure"].reset_index()


train[train["label"]=="pure"].reset_index(drop=True)


labels=train["label"].unique()
for lab in labels:
    trainl=train[train["label"]==lab].reset_index(drop=True)
    list_all=trainl["image"].tolist()
    idx=int(len(list_all)*0.7)
    list_train=list_all[:idx]
    list_validation=list_all[idx:]
    os.makedirs("CORN_DSI_23/train/"+lab,exist_ok=True)
    os.makedirs("CORN_DSI_23/validation/"+lab,exist_ok=True)
    for fname in tqdm.tqdm(list_train):
        fname=fname.split("/")[-1]
        src="/kaggle/input/kaggle-pog-series-s01e03/corn/train/"+fname
        dst="CORN_DSI_23/train/"+lab+"/"+fname
        shutil.copy(src,dst)
    for fname in tqdm.tqdm(list_validation):
        fname=fname.split("/")[-1]
        src="/kaggle/input/kaggle-pog-series-s01e03/corn/train/"+fname
        dst="CORN_DSI_23/validation/"+lab+"/"+fname
        shutil.copy(src,dst)

