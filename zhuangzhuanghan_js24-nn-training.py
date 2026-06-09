import os # move files
import pickle # dump/serilize models
######################
import polars as pl # lazy pipeline data
####################
import numpy as np
import pandas as pd
#####################
import warnings
warnings.filterwarnings("ignore")
#######################
import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers import WandbLogger
import wandb # w,b tracker
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class CFG():
    use_gpu = True
    gpu_id = 0
    seed = 42
    model = 'nn'
    use_wandb = False
    ##################
    loader_workers = 4
    batch_size = 8192
    ################# Key Params
    lr = 1e-3
    weight_decay = 5e-4
    dropouts = [0.1, 0.1]
    n_hidden = [512, 512, 256]
    patience = 25
    #################
    max_epochs = 2000
    n_fold = 5
###########
# CFG.n_fold


#####################################
class CustomDataset(Dataset):
    # this custom dataset will be called by the datamodule deifned below
    # 1. setup tensor for float values of features, labels,and weights
    #     - this will be used in NN defined later
    # 2. easy to split
    def __init__(self, df, accelerator):
        self.features = torch.FloatTensor(
            df[feature_names].values).to(accelerator)
        self.labels = torch.FloatTensor(
            df[label_name].values).to(accelerator)
        self.weights = torch.FloatTensor(
            df[weight_name].values).to(accelerator)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        w = self.weights[idx]
        return x, y, w
        
######################################
class DataModule(LightningDataModule): 
    # Here we use PytorchLightening DataModule
    # to preprocess; It's less verbose than Pytorch
    
    def __init__(
        self, train_df, batch_size, 
        valid_df=None, accelerator='cpu'):
        super().__init__()
        self.df = train_df
        self.batch_size = batch_size
        self.dates = self.df['date_id'].unique()
        self.accelerator = accelerator
        self.train_dataset = None
        self.valid_df = None
        if valid_df is not None:
            self.valid_df = valid_df
        self.val_dataset = None

    def setup(self, fold=0, N_fold=5, stage=None):
        # Split dataset
        selected_dates = [date for ii, date in enumerate(self.dates) if ii % N_fold != fold]
        df_train = self.df.loc[self.df['date_id'].isin(selected_dates)]
        self.train_dataset = CustomDataset(df_train, self.accelerator)
        if self.valid_df is not None:
            df_valid = self.valid_df
            self.val_dataset = CustomDataset(df_valid, self.accelerator)

    def train_dataloader(self, n_workers=0):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=n_workers)

    def val_dataloader(self, n_workers=0):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=n_workers)


# Custom R2 metric for validation
def r2_val(y_true, y_pred, sample_weight):
    nom = np.average((y_pred - y_true) ** 2, weights=sample_weight)
    denom = (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    r2 = 1 -  nom/denom 
    return r2


class NN(LightningModule):
    # use pytorch lightning
    
    def __init__(self, input_dim, hidden_dims, dropouts, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        layers = []
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.BatchNorm1d(in_dim))
            
            if i > 0: # skip first layer
                layers.append(nn.SiLU())
            
            if i < len(dropouts): # dropout the first few layers
                layers.append(nn.Dropout(dropouts[i]))
            
            layers.append(nn.Linear(in_dim, hidden_dim))
            # layers.append(nn.ReLU())
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, 1)) 
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_step_outputs = []

    def forward(self, x):
        return 5 * self.model(x).squeeze(-1)  

    def training_step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none') * w  #
        loss = loss.mean()
        self.log(
            'train_loss', loss, 
            on_step=False, on_epoch=True, 
            batch_size=x.size(0))
        return loss

    def validation_step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none') * w
        loss = loss.mean()
        self.log(
            'val_loss', loss, 
            on_step=False, on_epoch=True, 
            batch_size=x.size(0))
        self.validation_step_outputs.append((y_hat, y, w))
        return loss

    def on_validation_epoch_end(self):
        # Calculate validation WRMSE at each epoch end
        y = torch.cat(
            [x[1] for x in self.validation_step_outputs]
        ).cpu().numpy()
        
        if self.trainer.sanity_checking:
            # do not log the model if sanity_check is not working
            prob = torch.cat(
                [x[0] for x in self.validation_step_outputs]
            ).cpu().numpy()
        else:
            prob = torch.cat(
                [x[0] for x in self.validation_step_outputs]
            ).cpu().numpy()
            weights = torch.cat(
                [x[2] for x in self.validation_step_outputs]
            ).cpu().numpy()
            # r2_val
            val_r_square = r2_val(y, prob, weights)
            self.log(
                "val_r_square", val_r_square, 
                prog_bar=True, on_step=False, 
                on_epoch=True)
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, 
            weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', 
            factor=0.5, patience=5,
            verbose=True)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v \
            for k, v in self.trainer.logged_metrics.items()
        }
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted_metrics}")


# input_path = '/kaggle/input/js24-preprocessing-create-lags'
# df = pl.scan_parquet(f"{input_path}/training.parquet").select(
#     [pl.col("date_id")]).collect().to_pandas()
##############
# df[df.date_id > 1500]
# df.value_counts()


# Load data

input_path = '/kaggle/input/js24-preprocessing-create-lags'
# Here we use the lagged data from a kaggler
# Own preprocessed data can be used by uploading to input
#####################################
feature_names = [
    f"feature_{i:02d}" for i in range(79)] + [
        f"responder_{idx}_lag_1" for idx in range(9)]
label_name = 'responder_6'
weight_name = 'weight'
###########################
# Note: We only take samples with date_id > 1500 to have the RAM not overcharged
df = pl.scan_parquet(
    f"{input_path}/training.parquet").filter(
    pl.col("date_id") > 1500).collect().to_pandas()

valid = pl.scan_parquet(f"{input_path}/validation.parquet").filter(
    pl.col("date_id") > 1500).collect().to_pandas()

df = pd.concat([df, valid]).reset_index(drop=True)# A trick to boost LB from 0.0045->0.005

# Device
device = torch.device(
    f'cuda:{CFG.gpu_id}' if torch.cuda.is_available() and CFG.use_gpu else 'cpu'
)
accelerator = 'gpu' if torch.cuda.is_available() and CFG.use_gpu else 'cpu'
loader_device = 'cpu'


# Init DataModule

df[feature_names] = df[feature_names].fillna(
    method = 'ffill').fillna(0)
valid[feature_names] = valid[feature_names].fillna(
    method = 'ffill').fillna(0)
data_module = DataModule(
    df, batch_size = CFG.batch_size, 
    valid_df = valid, accelerator = loader_device
)


import gc
del df
gc.collect()
###############################
# RUNCV = True # Turn it on when re-train
RUNCV = False
###############################
if RUNCV:
    
    for fold in range(CFG.n_fold):
        data_module.setup(fold, CFG.n_fold)
        
        # Obtain input dimension
        
        input_dim = data_module.train_dataset.features.shape[1]
        
        
        # Initialize Model
        model = NN(
            input_dim = input_dim,
            hidden_dims = CFG.n_hidden,
            dropouts = CFG.dropouts,
            lr = CFG.lr,
            weight_decay = CFG.weight_decay
        )
        
        # Initialize Callbacks
        
        early_stopping = EarlyStopping(
            'val_loss', 
            patience=CFG.patience, 
            mode='min', verbose=False
        )
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss', 
            mode='min', 
            save_top_k=1, 
            verbose=False, 
            filename=f"./models/nn_{fold}.model"
        ) 
        timer = Timer()
        
        # Initialize Trainer
        
        trainer = Trainer(
            max_epochs=CFG.max_epochs,
            accelerator=accelerator,
            devices="auto" if CFG.use_gpu else None,
            logger=None,
            callbacks=[early_stopping, checkpoint_callback, timer],
            enable_progress_bar=True
        )
        
        # Start Training
        
        trainer.fit(
            model, data_module.train_dataloader(CFG.loader_workers), 
            data_module.val_dataloader(CFG.loader_workers)
        )
        
        # You can find trained best model in your local path
        print(f'Fold-{fold} Training completed in {timer.time_elapsed("train"):.2f}s')

