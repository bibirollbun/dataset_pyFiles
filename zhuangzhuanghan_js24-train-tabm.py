import os
import pickle
import joblib
from tqdm import tqdm
##################
import polars as pl
import numpy as np
import pandas as pd
################


############################
# Turn This on to Download TabM (if not download bf)
!git clone https://github.com/yandex-research/tabm
!pip install rtdl_num_embeddings


import sys
sys.path.append('root/')
#############################
import warnings
warnings.filterwarnings("ignore")
####################
# PyTorch 
import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
########################
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
######################
# Note: 'tabm_refernce' is a py script to funnel
# the main TabM structure to a pre-defined PyTorch Model class
# but leave some key parameters to be tuned
# see 'github.com/yandex-research/tabm' for more details
from tabm_reference import Model, make_parameter_groups
#####################


# Convert the scatter integers to consecutive
category_mappings = {
    'feature_09': {
        2: 0, 4: 1, 9: 2, 11: 3, 12: 4, 14: 5, 
        15: 6, 25: 7, 26: 8, 30: 9, 34: 10, 42: 11, 
        44: 12, 46: 13, 49: 14, 50: 15, 57: 16, 64: 17, 
        68: 18, 70: 19, 81: 20, 82: 21},
    'feature_10': {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 
        6: 5, 7: 6, 10: 7, 12: 8},
    'feature_11': {
        9: 0, 11: 1, 13: 2, 16: 3, 24: 4, 25: 5, 34: 6, 
        40: 7, 48: 8, 50: 9, 59: 10, 62: 11, 63: 12, 66: 13,
        76: 14, 150: 15, 158: 16, 159: 17, 171: 18, 195: 19, 
        214: 20, 230: 21, 261: 22, 297: 23, 336: 24, 
        376: 25, 388: 26, 410: 27, 522: 28, 534: 29, 539: 30},
    'symbol_id': {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 
        10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 
        17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 
        24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31,
        32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38},
    'time_id' : {i : i for i in range(968)}}


def encode_column(df, column, mapping):
    max_value = max(mapping.values())  

    def encode_category(category):
        return mapping.get(category, max_value + 1)  
    
    return df.with_columns(
        pl.col(column).map_elements(encode_category, return_dtype=pl.Int16).alias(column)
    )

# Define feature, label, and weight names

feature_names = ['time_id', 'symbol_id', 'responder_6', 'responder_3', 'weight'
                ] + [f"feature_{i:02d}" for i in range(79) if i != 61] + [
                    f"responder_{idx}_lag_1" for idx in range(9)]

label_name = 'responder_6'
weight_name = 'weight'
feature_cat = ['feature_09', 'feature_10', 'feature_11', 'symbol_id', 'time_id']
feature_cont = [
    col for col in feature_names if col not in feature_cat + [
        label_name, weight_name]
]

feature_cont_idx =  [feature_names.index(col) for col in feature_cont]
feature_cat_idx = [feature_names.index(col) for col in feature_cat]
label_idx = feature_names.index(label_name)
weight_idx = feature_names.index(weight_name)

# Use the most voted Lag-1 preprocessed dataset from Kaggle Discussions

input_path = '/kaggle/input/js24-preprocessing-create-lags'
train_original = pl.scan_parquet(
    f"{input_path}/training.parquet").sort(['date_id', 'time_id', 'symbol_id'])
valid_original = pl.scan_parquet(
    f"{input_path}/validation.parquet").sort(['date_id', 'time_id', 'symbol_id'])

for col in feature_cat:
    train_original = encode_column(train_original, col, category_mappings[col])
    valid_original = encode_column(valid_original, col, category_mappings[col])


# Load Data and Split

USE_FULLSIZE = False 
# for test runs | Turn this on for actual training
# kaggle env may not have enough RAM for full training, 
# pls use private server
DATE_AF = 1500

if USE_FULLSIZE:
    train_original = pl.concat([train_original, valid_original])
    df = train_original\
        .filter(pl.col('date_id') >= 252)\
        .select(feature_names)\
        .collect().to_numpy()
    df[np.isnan(df)] = 0
    valid = None
else:
    df = train_original\
        .filter(pl.col('date_id') >= DATE_AF)\
        .select(feature_names)\
        .collect().to_numpy()
    df[np.isnan(df)] = 0
    
    valid = valid_original\
        .select(feature_names)\
        .collect().to_numpy()
    valid[np.isnan(valid)] = 0


import pytorch_lightning as pl 
# pl is taken by polars previously
# now is changed to alias pytorch_lightning


class CFG():
    use_gpu = True 
    # To finish in reasonale time, pls use GPU
    # use GPU P100
    gpu_id = 0
    seed = 42
    loader_workers = 12   
    batch_size = 8192
    lr = 1e-3
    weight_decay = 8e-4
    n_cont_features = len(feature_cont)
    n_cat_features = 5
    cat_cardinalities = None if feature_cat is None else  [23, 10, 32, 40, 969]
    max_epochs = 7


# Define Custom DataSet

class CustomDataset(Dataset):
    def __init__(self, array):
        self.features_cont = torch.FloatTensor(array[:, feature_cont_idx])
        self.features_cat = torch.LongTensor(array[:, feature_cat_idx])
        self.labels = torch.FloatTensor(array[:, label_idx] )
        self.weights = torch.FloatTensor(array[:, weight_idx])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x_cont = self.features_cont[idx]
        x_cat = self.features_cat[idx]
        y = self.labels[idx]
        w = self.weights[idx]
        return x_cont, x_cat, y, w


# Define TabM Model Specs

class NN(LightningModule):
    def __init__(self, n_cont_features, cat_cardinalities, n_classes, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.k = 16
        self.model = Model(
                n_num_features=n_cont_features,
                cat_cardinalities=cat_cardinalities,
                n_classes=n_classes,
                backbone={
                    'type': 'MLP',
                    'n_blocks': 3 ,
                    'd_block': [512, 512, 512],
                    'dropout': 0.25 ,
                },
                bins=None,
                num_embeddings= None,
                arch_type='tabm',
                k=self.k,
            )

        self.lr = lr
        self.weight_decay = weight_decay
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.n_classes = n_classes
        self.loss_fn = R2Loss()
        # self.loss_fn = nn.MSELoss()

    def forward(self, x_cont, x_cat):
        return self.model(x_cont, x_cat).squeeze(-1)

    def training_step(self, batch):
        x_cont,x_cat, y, w= batch
        x_cont = x_cont + torch.randn_like(x_cont) * 0.02
        y_hat = self(x_cont, x_cat)

        if self.n_classes == 1:
            loss = self.loss_fn(y_hat.flatten(0, 1), 
                                y.repeat_interleave(self.k))
            self.training_step_outputs.append((y_hat.mean(1), y, w))
        else:
            loss = self.loss_fn(y_hat[:, :, 0].flatten(0, 1), 
                                 y.repeat_interleave(self.k))
            self.training_step_outputs.append((y_hat[:, :, 0].mean(1), y, w))

        self.log(
            'train_loss', loss, on_step=True, on_epoch=True, 
            prog_bar=True, logger=True, batch_size=x_cont.size(0)
        )
        return loss

    def validation_step(self, batch):
        x_cont,x_cat, y, w = batch
        if len(x_cat.size()) == 1:
            x_cat = None
        # x_cont = x_cont + torch.randn_like(x_cont) * 0.02
        y_hat = self(x_cont, x_cat)

        if self.n_classes == 1:
            loss = self.loss_fn(y_hat.flatten(0, 1), 
                                y.repeat_interleave(self.k))
            self.validation_step_outputs.append((y_hat.mean(1), y, w))
        else:
            loss = self.loss_fn(y_hat[:, :, 0].flatten(0, 1), 
                                 y.repeat_interleave(self.k))
            self.validation_step_outputs.append((y_hat[:, :, 0].mean(1), y, w))

        self.log(
            'val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, 
            logger=True, batch_size=x_cont.size(0))
        return loss

    def on_validation_epoch_end(self):
        y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
        
        if self.trainer.sanity_checking:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
        else:
            prob = torch.cat(
                [x[0] for x in self.validation_step_outputs]).cpu().numpy()
            weights = torch.cat(
                [x[2] for x in self.validation_step_outputs]).cpu().numpy()
            # r2_val
            val_r_square = r2_val(y, prob, weights)

            val_r_square_adj = 1 - (1 - r2_score(y, prob)) * (len(y) - 1) / (len(y) - 1 - 16)
            self.log("val_r_square", val_r_square, prog_bar=True, 
                     on_step=False, on_epoch=True)
            self.log("val_r_square_adj", val_r_square_adj, prog_bar=True, 
                     on_step=False, on_epoch=True)
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            make_parameter_groups(self.model), 
            lr=self.lr, 
            weight_decay=self.weight_decay)

        return {
            'optimizer': optimizer,
        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        y = torch.cat([x[1] for x in self.training_step_outputs]).cpu().numpy()
        prob = torch.cat([x[0] for x in self.training_step_outputs]).detach().cpu().numpy()
        weights = torch.cat([x[2] for x in self.training_step_outputs]).cpu().numpy()
        # r2_training
        train_r_square = r2_val(y, prob, weights)
        train_r_square_adj = 1 - (1 - r2_score(y, prob)) * (len(y) - 1) / (len(y) - 1 - 16)
        # self.log("train_r_square", train_r_square, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_r_square_adj", train_r_square_adj, prog_bar=True, 
                 on_step=False, on_epoch=True)
        self.training_step_outputs.clear()

        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
        formatted_metrics = {k: f"{v:.7f}" for k, v in metrics.items()}

        formatted_metrics.pop('train_loss_step', None)
        print(f"Epoch {epoch}: {formatted_metrics}")


# Loss Func

def r2_val(y_true, y_pred, sample_weight):
    residuals = sample_weight * (y_true - y_pred) ** 2
    nom = np.sum(residuals)
    denom = np.sum(sample_weight * (y_true) ** 2)
    r2 = 1 - nom/denom
    return r2


class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        mse_loss = torch.sum((y_pred - y_true) ** 2)
        var_y = torch.sum(y_true ** 2)
        loss = mse_loss / (var_y + 1e-38)
        return loss


# Clean the RAM if NEEDED

# import gc
# ###########
# if 'df' in globals():
#     del df
# ###########
# gc.collect()


# Non-lazy/Direct Load Data

####### Device Select
device = torch.device(
    f'cuda:{CFG.gpu_id}' if torch.cuda.is_available() and CFG.use_gpu else 'cpu')
accelerator = 'gpu' if torch.cuda.is_available() and CFG.use_gpu else 'cpu'
loader_device = 'cpu'

#######
train_ds = CustomDataset(df)
train_dl = DataLoader(
    train_ds, batch_size = CFG.batch_size, 
    shuffle=True, num_workers = CFG.loader_workers)

if valid is not None:
    valid_ds = CustomDataset(valid)
    valid_dl = DataLoader(
        valid_ds, batch_size = CFG.batch_size, 
        shuffle=True, num_workers = CFG.loader_workers)


# Init Model

model = NN(
    n_cont_features = CFG.n_cont_features,
    cat_cardinalities = CFG.cat_cardinalities,
    n_classes = 1,
    lr = CFG.lr,
    weight_decay = CFG.weight_decay
)

# Init Callbacks

checkpoint_callback = ModelCheckpoint(
    monitor = 'val_r_square', mode = 'max', save_top_k = 1, 
    verbose = False, filename = f"./models/tabm.model") 

every_heckpoint_callback = ModelCheckpoint(
    every_n_epochs = 1, save_top_k = -1, verbose = False, 
    filename = "./models/tabm_{epoch:02d}") 

timer = Timer()


# Init PyTorchLightning Trainer

print("Training Epoch is ", CFG.max_epochs)
trainer = Trainer(
    default_root_dir = 'root/',
    max_epochs = CFG.max_epochs,
    accelerator = accelerator,
    devices = [CFG.gpu_id] if CFG.use_gpu else 'auto',
    # devices = 'auto'
    callbacks = [ checkpoint_callback, every_heckpoint_callback, timer],
    enable_progress_bar = True,
    val_check_interval = 0.5,
)
# Start Training
trainer.fit(model, train_dl, valid_dl)

print(f'\Training completed in {timer.time_elapsed("train"):.2f}s')

