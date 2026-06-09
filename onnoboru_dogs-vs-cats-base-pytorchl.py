!pip install lightning


import os
import gc
import re
import sys
import time
import copy
import random 
import glob 
import zipfile
import shutil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl  
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2   
from PIL import Image

import transformers

from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class CFG :
    debug_one_epoch = True
    debug_one_fold = False
    only_infer = False
    num_workers = 16
    batch_size = 64
    num_epochs = 10
    lr = 1e-3
    early_stopping_round = 5
    warmup_prop = 0.1
    random_seed = 42
    n_splits = 5
    model_name = "resnet18" # timm で使うモデル名
    pretrained_path = None
    train_dir = None # 学習データセットのパス
    test_dir = None # テストデータセットのパス
    optimizer = torch.optim.AdamW
    criterion = nn.BCEWithLogitsLoss()
    scheduler = transformers.get_linear_schedule_with_warmup
    input_imgsize = 224
    data_dir = "../input/dogs-vs-cats-redux-kernels-edition/"
    kaggle_working_dir = "/kaggle/working/"
    
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(CFG.random_seed)

if  CFG.debug_one_epoch :
    CFG.num_epochs = 1

print('KAGGLE_URL_BASE' in set(os.environ.keys()))



submission = pd.read_csv(os.path.join(CFG.data_dir, "sample_submission.csv"))
if 'KAGGLE_URL_BASE' in set(os.environ.keys()) :
    kaggle_train_dir = os.path.join(CFG.kaggle_working_dir, "train")
    # すでに解凍されている場合は解凍しない
    if not os.path.exists(kaggle_train_dir) :
        shutil.unpack_archive(os.path.join(CFG.data_dir, "train.zip"), CFG.kaggle_working_dir)
    
    kaggle_test_dir = os.path.join(CFG.kaggle_working_dir, "test")
    if not os.path.exists(kaggle_test_dir) :
        shutil.unpack_archive(os.path.join(CFG.data_dir, "test.zip"), CFG.kaggle_working_dir)
        
    CFG.data_dir = CFG.kaggle_working_dir
    
CFG.train_dir = os.path.join(CFG.data_dir, "train")
CFG.test_dir = os.path.join(CFG.data_dir, "test")

train_list = glob.glob(os.path.join(CFG.data_dir, "train", "*.jpg"))
test_list = glob.glob(os.path.join(CFG.data_dir, "test", "*.jpg"))


train_df = pd.DataFrame(train_list, columns=["path"])
train_df["class"] = train_df["path"].apply(lambda x : x.split("/")[-1].split(".")[0])
train_df["class"] = train_df["class"].map({"dog" : 1, "cat" : 0})
test_df = pd.DataFrame(test_list, columns=["path"])
test_df["class"] = -1
# test_df に対しては path の数字が昇順であることを保証するために id を追加
test_df["id"] = test_df["path"].apply(lambda x : int(x.split("/")[-1].split(".")[0]))
test_df = test_df.sort_values("id").reset_index(drop=True)


train_transform = A.Compose([
    A.Resize(CFG.input_imgsize, CFG.input_imgsize),
    A.HorizontalFlip(p=0.5), # 50% の確率で水平反転
    A.Normalize(),
    ToTensorV2()
])
test_transform = A.Compose([
    A.Resize(CFG.input_imgsize, CFG.input_imgsize),
    A.Normalize(),
    ToTensorV2()
])


class DogsCatsDataset(Dataset) :
    def __init__(self, df, transform=None) :
        self.df = df # さっきの pandas dataframe を受け取る
        self.transform = transform # 画像の変換処理を受け取る

    def __len__(self) :
        return len(self.df)
    
    def __getitem__(self, idx) :
        img = Image.open(self.df.iloc[idx, 0])
        img = self.transform(image = np.array(img))["image"]
        label = self.df.iloc[idx, 1].astype(np.float32)
        return img, label


class DogCatModel(nn.Module) :
    def __init__(self) :
        super(DogCatModel, self).__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        
    def forward(self, x) :
        return self.model(x)


class dog_vs_cats_pl_model(pl.LightningModule) :
    def __init__(self, model) :
        super(dog_vs_cats_pl_model, self).__init__()
        self.model = model
        self.criterion = CFG.criterion
        
    def forward(self, x) :
        return self.model(x)
    
    def training_step(self, batch, batch_idx) :
        img, label = batch
        output = self(img)
        loss = self.criterion(output.squeeze(-1), label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx) :
        img, label = batch
        output = self(img)
        loss = self.criterion(output.squeeze(-1), label)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx) :
        img, _ = batch
        output = self(img)
        output = torch.sigmoid(output).cpu().numpy()
        return output
        
    
    def configure_optimizers(self) :
        optimizer = CFG.optimizer(self.parameters(), lr=CFG.lr)
        num_training_steps = len(self.train_dataloader)*CFG.num_epochs
        num_warmup_steps = int(num_training_steps * CFG.warmup_prop)
        scheduler = {
            "scheduler" :  transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps),
            "interval" : "step",
            "frequency" : 1
        }
        
        return [optimizer], [scheduler]
    


def run_train_cv_pl(train, test):
    kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_seed)
    oof = np.zeros((len(train), 1)) 
    predictions =[]
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)) :
        print(f"====================fold : {fold}====================")
        train_df = train.iloc[train_idx].reset_index(drop=True)
        valid_df = train.iloc[valid_idx].reset_index(drop=True)
        
        train_dataset = DogsCatsDataset(train_df, transform=train_transform)
        valid_dataset = DogsCatsDataset(valid_df, transform=test_transform)
        test_dataset = DogsCatsDataset(test_df, transform=test_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
        
        model = DogCatModel()
        lightning_model = dog_vs_cats_pl_model(model)
        lightning_model.train_dataloader = train_loader
        lightning_model.valid_dataloader = valid_loader

        early_stopping = EarlyStopping(
            monitor="valid_loss",
            mode="min", 
            patience=CFG.early_stopping_round,
        )
        checkpoint = ModelCheckpoint(
            monitor="valid_loss", 
            mode="min", 
            dirpath="checkpoints", 
            filename=f"{CFG.model_name}_fold{fold}", 
            save_top_k=1,
        )
        seed_everything(CFG.random_seed)
        logger = pl.loggers.TensorBoardLogger("logs", name=f"{CFG.model_name}_fold{fold}")
        trainer = pl.Trainer(max_epochs=CFG.num_epochs, accelerator="gpu", precision=16, logger=logger, callbacks=[early_stopping, checkpoint])
        trainer.fit(lightning_model, train_loader, valid_loader)

        # best_model = DogCatModel()
        # best_model.load_state_dict(torch.load(os.path.join("checkpoints", f"{CFG.model_name}_fold{fold}.ckpt"))["state_dict"])
        # best_model.eval()
        
        valid_preds_list = trainer.predict(lightning_model, valid_loader)
        valid_preds_arr = np.concatenate(valid_preds_list)
        oof[valid_idx] = valid_preds_arr
        
        test_preds_list = trainer.predict(lightning_model, test_loader)
        test_preds_arr = np.concatenate(test_preds_list)
        predictions.append(test_preds_arr)
        
        del model, lightning_model, trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        if CFG.debug_one_fold :
            break
    
    predictions = np.mean(predictions, axis=0)
    
    return {
        "oof" : oof,
        "predictions" : predictions
    }


def main() :
    if CFG.only_infer :
        test_dataset = DogsCatsDataset(test_df, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
        model = DogCatModel()
        lightning_model = dog_vs_cats_pl_model(model)
        predictions = []
        for fold in range(CFG.n_splits) :
            model = lightning_model.load_from_checkpoint(f"checkpoints/{CFG.model_name}_fold{fold}.ckpt")
            trainer = pl.Trainer(accelerator="gpu", precision=16, logger=False)
            predictions.append(trainer.predict(model, test_loader))
        submission["label"] = np.mean(predictions, axis=0)
        submission.to_csv("submission.csv", index=False)
        
    else :
        result = run_train_cv_pl(train_df, test_df)
        oof_preds = result["oof"]
        predictions = result["predictions"]
        submission["label"] = predictions
        submission.to_csv("submission.csv", index=False)
        train_df["oof_preds"] = oof_preds   
        train_df.to_csv("oof_preds.csv", index=False)
        if CFG.debug_one_fold == False :
            print(f"oof log loss : {log_loss(train_df['class'], oof_preds)}")
        
if __name__ == "__main__" :
    main()

