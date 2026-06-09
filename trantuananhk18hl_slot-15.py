## Parameters
data_config = {
    'train.csv': '../input/tabular-playground-series-sep-2022/train.csv',
    'test.csv': '../input/tabular-playground-series-sep-2022/test.csv',
    'sample_submission.csv': '../input/tabular-playground-series-sep-2022/sample_submission.csv',
}

exp_config = {
    'competition_name': 'tps-sep-2022',
    'history_period': 60,  ## Lookback period (days)
    'horizon_period': 30,  ## Forecast period (days)
    'val_ratio': 0.2,  ## Train-vaild split ratio
    'batch_size': 512,
    'train_epochs': 10,
    'learning_rate': 5e-3,
    'gamma': 0.95,  ## parameter of learning scheduler
    'train_limit': True,  ## Use or not the data before 2020
    'checkpoint_filepath': './tmp/model/checkpoint.cpt',
    'finalize': True,  ## For the model finalization
    'finalize_epochs': 5,  ## None or int
    'finalized_filepath': './tmp/model/finalized.cpt',
}

model_config = {
    'emb_dim': 6,  ## categorical features' representation dim
    'n_blocks': 2,  ## number of N-BEATS blocks in a stack
    'n_stacks': 4,  ## number of N-BEATS stacks
    'width': 32,  ## hidden dim in N-BEATS
}

print('Parameters setted!')


## Import dependencies 
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt 
%matplotlib inline

import seaborn as sns
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os, sys, pathlib, gc
import re, math, random, time
import datetime as dt
from tqdm import tqdm
from typing import Optional, Union, Tuple
from collections import OrderedDict

import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

print('import done!')


## For reproducible results    
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
    print('Seeds setted!')
global_seed = 42
seed_all(global_seed)


## Limit GPU Memory in TensorFlow
## Because TensorFlow, by default, allocates the full amount of available GPU memory when it is launched. 
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'\
              .format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")
    
    
## For Seaborn Setting
custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    'grid.alpha': 0.3,
    'figure.figsize': (16, 6),
    'axes.titlesize': 'Large',
    'axes.labelsize': 'Large',
    'figure.facecolor': '#fdfcf6',
    'axes.facecolor': '#fdfcf6',
}
#cluster_colors = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
sns.set_theme(
    style='whitegrid',
    #palette=sns.color_palette(cluster_colors),
    rc=custom_params,
)


## Data Loading
train_df = pd.read_csv(data_config['train.csv'])
test_df = pd.read_csv(data_config['test.csv'])
submission_df = pd.read_csv(data_config['sample_submission.csv'])

print(f'train_length: {len(train_df)}')
print(f'test_lenght: {len(test_df)}')
print(f'submission_length: {len(submission_df)}')


## Null Value Check
print('train_df.info()'); print(train_df.info(), '\n')
print('test_df.info()'); print(test_df.info(), '\n')

## train_df Check
train_df.head()


## Features and Targets
original_features = ['date', 'country', 'store', 'product']
target = 'num_sold'

## Number of unique values in each features.
n_unique_features = {feature: train_df[feature].nunique() for feature in original_features}
n_unique_features


## test_df Check
test_df.head()


## Statistics of num_sold
train_df['num_sold'].describe()


## num_sold of each product in each country
ax = sns.barplot(data=train_df, x='product', y='num_sold', hue='country')


## Total sells by Country
train_df['date'] = pd.to_datetime(train_df['date'])

ax = sns.lineplot(
    data=train_df.groupby([
        train_df.date.dt.strftime('%Y-%m'),
        train_df.country
    ])['num_sold'].sum().reset_index(),
    x='date',
    y='num_sold',
    hue='country',
)

ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))


## Distrubutions of num_sold
fig, ax = plt.subplots(12, 4, figsize=(25, 50))
ax = ax.flatten()

for i, (combination, df) in enumerate(train_df.groupby(['country', 'store', 'product'])):
    sns.histplot(df.num_sold, ax=ax[i])
    ax[i].set_title(' | '.join(combination))
    
plt.tight_layout()


## Datetime Feature Engineering
def date_feature_eng(df, date_column_name, drop=True):
    df[date_column_name] = pd.to_datetime(df[date_column_name])
    df['year'] = df[date_column_name].dt.year
    df['month'] = df[date_column_name].dt.month
    df['day'] = df[date_column_name].dt.day
    df['dayofweek'] = df[date_column_name].dt.dayofweek
    
    if drop:
        df = df.drop(date_column_name, axis=1)
        
    return df

## Create date-related features
train_df = date_feature_eng(train_df, 'date')
test_df = date_feature_eng(test_df, 'date')

## Use or not the data before 2020
if exp_config['train_limit']:
    train_df = train_df[train_df['year']==2020]
    train_df = train_df.reset_index(drop=True)


## Standardization of numerical featurs and target

## Only 'num_sold' is numerical in this data set
train_num_sold = train_df['num_sold'].values
train_num_sold = train_num_sold.reshape(-1, 1)

## Using sklearn.preprocessing.StandardScaler
sc = StandardScaler()
sc.fit(train_num_sold)
print('StandardScaler mean: ', sc.mean_)
print('StandardScaler scale: ', sc.scale_)

## Check
train_df['num_sold'] = sc.transform(train_num_sold)
train_df['num_sold'].describe()


## Making Lookup table of categorical featurs and target
numerical_columns = ['num_sold']
categorical_columns = ['country', 'store', 'product', 'month', 'day', 'dayofweek']
## I will not use 'year' as a categorical feature.
## Because there are no overlapping in 'year' feature between train and test data.
## It means that the model can't learn the embedding of year==2021.

## Using sklearn.preprocessing.OrdinalEncoder
oe = OrdinalEncoder(handle_unknown='error',
                    dtype=np.int64)
encoded = oe.fit_transform(train_df[categorical_columns].values)
#decoded = oe.inverse_transform(encoded)
train_df[categorical_columns] = encoded
test_df[categorical_columns] = oe.transform(test_df[categorical_columns].values)

## Check
encoder_categories = oe.categories_
encoder_categories


train_data = train_df.copy()

## Settings
n_country = 6
n_store = 2
n_product = 4
n_items = n_country * n_store * n_product ## 48
n_days = int(len(train_data) / n_items) ## 1461

history_period = exp_config['history_period']  ## Lookback period
horizon_period = exp_config['horizon_period']  ## Forecast period
before_start_idx = n_items * history_period
after_end_idx = n_items * horizon_period

## Helper Functions
def collect_past_data(idx, period, n_items) -> list:
    if idx < n_items * period:
        return []
    past_start_idx = idx - n_items * period
    past_data_ids = [i * n_items + past_start_idx for i in range(period)]
    return past_data_ids

def collect_future_data(idx, period, n_items) -> list:
    future_data_ids = [i * n_items + idx for i in range(period)]
    return future_data_ids

## Operation Check
a = 10000
b = collect_past_data(a, history_period, n_items)
c = collect_future_data(a, horizon_period, n_items)
print('index example: ', a, '\n')
print('history index: \n', b, '\n')
print('prediction index: \n', c, '\n')


## Dataset
class TPSSep22TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, numerical_columns, 
                 categorical_columns,
                 history_period=30,
                 horizon_period=30,
                 n_items=48,
                 target=None):
        self.df = df
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.history_period = history_period
        self.horizon_period = horizon_period
        self.n_items = n_items
        self.target = target
        self.before_start_idx = n_items * history_period
        self.after_end_idx = n_items * horizon_period
        
    def __len__(self):
        return (len(self.df) - self.before_start_idx - self.after_end_idx)
    
    def __getitem__(self, index):
        data = OrderedDict()
        index = index + self.before_start_idx
        
        data['row_id'] = self.df['row_id'][index]
        
        for nc in self.numerical_columns:
            past_data_ids = collect_past_data(
                index,
                self.history_period,
                self.n_items
            )
            x = torch.tensor(
                self.df[nc][past_data_ids].values,
                dtype=torch.float32
            )
            name = 'past_' + nc
            data[name] = x
        
        for cc in self.categorical_columns:
            x = torch.tensor(
                self.df[cc][index],
                dtype=torch.int32
            )
            x = torch.unsqueeze(x, dim=0)
            data[cc] = x
        
        if self.target is not None:
            if index + self.after_end_idx < len(self.df):
                future_data_ids = collect_future_data(
                    index,
                    self.horizon_period,
                    self.n_items
                )
                label = torch.tensor(
                    self.df[self.target][future_data_ids].values,
                    dtype=torch.float32
                )
            else:
                label = torch.tensor(
                    self.df[self.target][index],
                    dtype=torch.float32
                )
            return data, label
        else:
            return data


## train-valid split
if exp_config['train_limit']:
    val_ratio = 0.3
else:
    val_ratio = exp_config['val_ratio']

n_val = int((len(train_data) - before_start_idx - after_end_idx) / n_items * val_ratio) * n_items
n_train = len(train_data) - before_start_idx - after_end_idx - n_val
print(n_train, n_val)

train = train_data[:n_train + before_start_idx].reset_index(drop=True)
valid = train_data[-(n_val + after_end_idx):].reset_index(drop=True)
print(len(train), len(valid))


## Making Dataset
train_ds = TPSSep22TrainDataset(
    train, 
    numerical_columns,
    categorical_columns,
    history_period,
    horizon_period,
    n_items,
    target
)

val_ds = TPSSep22TrainDataset(
    valid,
    numerical_columns,
    categorical_columns,
    history_period,
    horizon_period,
    n_items,
    target
)

## Operation Check
print('length of train_ds: ', len(train_ds))
print('length of val_ds: ', len(val_ds))
index = 0
sample = train_ds.__getitem__(index)
print('\n', sample)


## Making DataLoader
batch_size =exp_config['batch_size']

train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
)

val_dl = torch.utils.data.DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
)

dl_dict = {'train': train_dl, 'val': val_dl}

## Operation Check
for dl_sample in (train_dl):
    break  
x_sample = dl_sample[0]
y_sample = dl_sample[1]
print('input keys: ', x_sample.keys())
print('label shape: ', y_sample.shape)


class Preprocessor(nn.Module):
    def __init__(self, numerical_features,
                 categorical_features, 
                 encoder_categories, emb_dim):
        super().__init__()
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.encoder_categories = encoder_categories
        self.emb_dim = emb_dim
        self.embed_layers = nn.ModuleDict()
        
        for i, categorical in enumerate(categorical_features):
            embedding = nn.Embedding(num_embeddings=len(encoder_categories[i]),
                                     embedding_dim=self.emb_dim,)
            self.embed_layers[categorical] = embedding
            
    def forward(self, x):
        x_nums = []
        for numerical in self.numerical_features:
            x_num = x[numerical]
            x_nums.append(x_num)
        if len(x_nums) > 0:
            x_nums = torch.cat(x_nums, dim=1)
        else:
            x_nums = torch.tensor(x_nums, dtype=torch.float32)
        
        x_cats = []
        for categorical in self.categorical_features:
            x_cat = self.embed_layers[categorical](x[categorical])
            x_cats.append(x_cat)
        if len(x_cats) > 0:
            x_cats = torch.cat(x_cats, dim=1)
        else:
            x_cats = torch.tensor(x_cats, dtype=torch.float32)
            
        return x_nums, x_cats


class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, output_dim, width):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, width)
        self.fc_b = nn.Linear(width, width, bias=False)
        self.fc_f = nn.Linear(width, width, bias=False)
        self.g_b = nn.Linear(width, input_dim)
        self.g_f = nn.Linear(width, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        theta_b = self.fc_b(x)
        theta_f = self.fc_f(x)
        
        backcast = self.g_b(theta_b)
        forecast = self.g_f(theta_f)
        
        return backcast, forecast


class NBeatsStack(nn.Module):
    def __init__(self, n_blocks, input_dim, output_dim, width):
        super().__init__()
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = NBeatsBlock(input_dim, output_dim, width)
            self.blocks.append(block)
            
    def forward(self, x):
        stack_forecast = []
        for i in range(self.n_blocks):
            backcast, forecast = self.blocks[i](x)
            x = x - backcast
            stack_forecast.append(forecast)
        stack_forecast = torch.stack(stack_forecast, axis=-1)
        stack_forecast = torch.sum(stack_forecast, axis=-1)
        stack_residual = x
        return stack_residual, stack_forecast


class NBeatsModel(nn.Module):
    def __init__(self, n_blocks, n_stacks, 
                 input_dim, output_dim, width):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        
        self.stacks = nn.ModuleList()
        for _ in range(n_stacks):
            stack = NBeatsStack(n_blocks, input_dim, output_dim, width)
            self.stacks.append(stack)
            
    def forward(self, x):
        global_forecast = []
        for i in range(self.n_stacks):
            stack_residual, stack_forecast = self.stacks[i](x)
            x = stack_residual
            global_forecast.append(stack_forecast)
        global_forecast = torch.stack(global_forecast, axis=-1)
        global_forecast = torch.sum(global_forecast, axis=-1)
        return global_forecast


## settings for models
emb_dim = model_config['emb_dim']
n_blocks = model_config['n_blocks']
n_stacks = model_config['n_stacks']
input_dim = exp_config['history_period']
output_dim = exp_config['horizon_period']
width = model_config['width']
num_epochs = exp_config['train_epochs']

numerical_features = ['past_num_sold']
categorical_features = ['country', 'store', 'product', 'month', 'day', 'dayofweek']

## Building Models
preprocessor = Preprocessor(
    numerical_features,
    categorical_features,
    encoder_categories,
    emb_dim
)

model = NBeatsModel(
    n_blocks,
    n_stacks,
    input_dim,
    output_dim,
    width
)

## Operation, Parameters and Model Structure Check
x_nums, x_cats = preprocessor(x_sample)
y = model(x_nums)
print('Input shape: ', x_nums.shape)
print('Output shape: ', y.shape)

print('# of Preprocessor parameters: ',\
      sum(p.numel() for p in preprocessor.parameters() if p.requires_grad))
print('# of N-BEATS parameters: ',\
      sum(p.numel() for p in model.parameters() if p.requires_grad))

model


## Loss Function
criterion = nn.MSELoss()

## Optimizer and Learning Rate Scheduler
learning_rate = exp_config['learning_rate']
gamma = exp_config['gamma']
steps_per_epoch = len(train) // batch_size

params = list(preprocessor.parameters()) + list(model.parameters())
optimizer = torch.optim.Adam(
    params=params,
    lr=learning_rate
)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
#                                                          T_max=num_epochs*steps_per_epoch)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=gamma
)


## Model Save & Load
cpt_filepath = exp_config['checkpoint_filepath']

## Function for Saving Model
def model_save(model, preprocessor,
               optimizer, scheduler, path):
    directory = path.split('/')[:-1]
    directory = '/'.join(directory)
    os.makedirs(directory, exist_ok=True)
    
    ## When you use multi GPUs with DataParallel
    model_to_save = model.module if hasattr(model, "module") else model
    
    checkpoint = {
        "model": model_to_save.state_dict(),
        "preprocessor": preprocessor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        cuda_random_state = {
            "cuda_random": torch.cuda.get_rng_state(),
            "cuda_random_all": torch.cuda.get_rng_state_all(),
        }
        checkpoint.update(cuda_random_state)
        
    torch.save(checkpoint, path)
    print('Model saved!')

    
## Function for Loading Model
def model_load(model, preprocessor,
               optimizer, scheduler, path):
    checkpoint = torch.load(path) 
    
    ## When you use multi GPUs with DataParallel
    if hasattr(model, "module"):  
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])
        
    preprocessor.load_state_dict(checkpoint["preprocessor"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    random.setstate(checkpoint["random"])
    np.random.set_state(checkpoint["np_random"])
    torch.set_rng_state(checkpoint["torch"])
    torch.random.set_rng_state(checkpoint["torch_random"])
    
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint["cuda_random"])
        torch.cuda.torch.cuda.set_rng_state_all(checkpoint["cuda_random_all"])
        
    print('Model loaded!')


## Function for the Model Training
def train_model(model, preprocessor, dl_dict,
                criterion, optimizer, lr_scheduler,
                num_epochs, cpt_filepath=None,
                finalize=False):
    
    ## Checking usability of GUP
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print('-------Start Training-------')
    model.to(device)
    ## We use preprocessor on CPU
    
    ## training and validation loop
    if finalize:
        phases = ['train']
    else:
        phases = ['train', 'val']
        
    losses = {phase: [] for phase in phases}
    best_val_loss = 100.
    best_epoch = 1
    for epoch in range(num_epochs):
        for phase in phases:
            if phase == 'train':
                preprocessor.train()
                model.train()
            else:
                preprocessor.eval()
                model.eval()
                
            epoch_loss = 0.0 
            
            for data, labels in tqdm(dl_dict[phase]):
                x_nums, x_cats = preprocessor(data)
                
                x_nums = x_nums.to(device)
                labels = labels.to(device)
                
                ## Optimizer Initialization
                optimizer.zero_grad()
                
                ## Forward Processing
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(x_nums)
                    loss = criterion(outputs, labels)
                    
                    ## Backward Processing and Optimization
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        
                    epoch_loss += loss.item() * x_nums.size(0)
            
            epoch_loss = epoch_loss / len(dl_dict[phase].dataset)
            losses[phase].append(epoch_loss)
            
            ## Saving the best Model
            if phase == 'val':
                if cpt_filepath is not None:
                    if epoch_loss < best_val_loss:
                        best_epoch = epoch + 1
                        best_val_loss = epoch_loss
                        model_save(
                            model, preprocessor,
                            optimizer, lr_scheduler,
                            cpt_filepath
                        )
            
            ## Displaying results
            print('Epoch {}/{} | {:^5} |  Loss: {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss ))
            
    if finalize:
    ## Saving finalized Model
        if cpt_filepath is not None:
            model_save(
                model, preprocessor, 
                optimizer, lr_scheduler,
                cpt_filepath
            )
            
    return model, preprocessor, losses, best_epoch


## Function for Plotting Losses
def plot_losses(losses, title=None):
    plt.figure(figsize=(7, 5))
    losses = pd.DataFrame(losses)
    losses.index = [i+1 for i in range(len(losses))]
    ax = sns.lineplot(data=losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title(title)


## Training
num_epochs = exp_config['train_epochs']
cpt_filepath = exp_config['checkpoint_filepath']

model, preprocessor, losses, best_epoch = train_model(
    model,
    preprocessor,
    dl_dict,
    criterion,
    optimizer,
    lr_scheduler,
    num_epochs,
    cpt_filepath
)

## Load the Best Model
model_load(
    model,
    preprocessor,
    optimizer,
    lr_scheduler,
    cpt_filepath
)

## Plot Losses
plot_losses(losses)


## Finalizing
if exp_config['finalize']:
    
    ## Setting Finalize Epochs
    if exp_config['finalize_epochs'] is not None:
        num_epochs = exp_config['finalize_epochs']
    else:
        num_epochs = best_epoch
    
    ## Making Dataset and DataLoader for Finalizing
    train_all_ds = TPSSep22TrainDataset(
        train_data, 
        numerical_columns,
        categorical_columns,
        history_period,
        horizon_period,
        n_items,
        target
    )
    
    train_all_dl = torch.utils.data.DataLoader(
        train_all_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    
    finalize_dl_dict = {'train': train_all_dl}
    
    ## Building Models
    preprocessor = Preprocessor(
        numerical_features,
        categorical_features,
        encoder_categories,
        emb_dim
    )
    
    model = NBeatsModel(
        n_blocks,
        n_stacks,
        input_dim,
        output_dim,
        width
    )
    
    ## Loss Function
    criterion = nn.MSELoss()
    
    ## Optimizer and Learning Rate Scheduler
    learning_rate = exp_config['learning_rate']
    gamma = exp_config['gamma']
    params = list(preprocessor.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(
        params=params,
        lr=learning_rate
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=gamma
    )
    
    ## Model Training
    finalized_filepath = exp_config['finalized_filepath']
    
    model, preprocessor, losses, _ = train_model(
        model,
        preprocessor,
        finalize_dl_dict,
        criterion,
        optimizer,
        lr_scheduler,
        num_epochs,
        finalized_filepath,
        finalize=True
    )
    
    ## Plot losses
    plot_losses(losses)


## Dataset for Test Data
class TPSSep22TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_df, train_df,
                 numerical_columns, 
                 categorical_columns,
                 history_period=30,
                 horizon_period=30,
                 n_items=48):
        self.train_df = train_df
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.history_period = history_period
        self.horizon_period = horizon_period
        self.n_items = n_items
        self.before_start_idx = n_items * history_period
        self.after_end_idx = n_items * horizon_period
        
        last_train_data = self.train_df.iloc[-self.before_start_idx:]
        self.test_df = pd.concat([last_train_data, test_df], axis=0, ignore_index=True)
        
    def __len__(self):
        return (len(self.test_df) - self.before_start_idx) // self.after_end_idx + 1
    
    def __getitem__(self, index):
        data = {}
        index = (index * self.after_end_idx + self.before_start_idx)
        
        for i in range(self.n_items):
            item_index = index + i
            data[str(i)] = {}
            data[str(i)]['row_id'] = item_index
            
            past_data_ids = collect_past_data(item_index,
                                              self.history_period,
                                              self.n_items)
            x = torch.tensor(self.test_df['num_sold'][past_data_ids].values,
                             dtype=torch.float32)
            data[str(i)]['history'] = x

        return data
        
    def update(self, new_data, column, row_id):
        row_ids = collect_future_data(row_id,
                                      self.horizon_period,
                                      self.n_items)
        max_id = self.test_df.iloc[-1]['row_id']
        if row_ids[-1] <= max_id:
            self.test_df.loc[self.test_df['row_id'].isin(row_ids), column] = new_data
        else:
            row_ids = np.array(row_ids)
            max_len = (row_ids <= max_id).sum()
            row_ids = row_ids[:max_len]
            
            row_ids = row_ids.tolist()
            new_data = new_data[:max_len]
            self.test_df.loc[self.test_df['row_id'].isin(row_ids), column] = new_data


## Making Test Dataset
test_data = test_df.copy()
test_ds = TPSSep22TestDataset(
    test_data,
    train_data,
    numerical_columns,
    categorical_columns,
    history_period,
    horizon_period,
    n_items
)

## Making Test DataLoader
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    drop_last=False
)

## Operation Check
print('length of test_ds: ', len(test_ds), '\n')
index = 0
test_sample = test_ds.__getitem__(index)
print('number of keys in test data: ', len(test_sample.keys()))
for key in test_sample.keys():
    print('row_id sample: ', test_sample[key]['row_id'])
    print('history sample shape: ', test_sample[key]['history'].shape)
    break


## Function for the Test Data Prediction
def model_predict(model, preprocessor, test_dl):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print('-------Start Prediction-------')
    model.to(device)
    
    preprocessor.eval()
    model.eval()
    
    for data_set in tqdm(test_dl):
        for i in range(n_items):
            data = data_set[str(i)]
            row_id = data['row_id'].item()
            row_id = row_id + test_dl.dataset.test_df['row_id'][0]
            x_nums = data['history']
            x_nums = x_nums.to(device)
            
            with torch.no_grad():
                outputs = model(x_nums)
                outputs = torch.squeeze(outputs)
                outputs = outputs.to('cpu').detach().numpy().copy()
                
            test_dl.dataset.update(outputs, 'num_sold', row_id)
            
    outputs = test_dl.dataset.test_df['num_sold']
    return outputs


## Prediction
outputs = model_predict(model, preprocessor, test_dl)
preds_normed = outputs.iloc[test_dl.dataset.before_start_idx:]

## post-processing
preds = (preds_normed * sc.scale_) + sc.mean_
submission_df['num_sold'] = preds.values
submission_df.to_csv('submission_cv.csv', index=False)

## Check
print('The number of null values: \n', submission_df.isnull().sum())
submission_df.head(10)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def load_data(col=None, path="/kaggle/input/energy-consumption-generation-prices-and-weather/energy_dataset.csv", verbose=False):
    df = pd.read_csv(path)
    if col is not None:
        df = df[col]
    if verbose:
        print(df.head())
    return df

print("Multivariate Sample")
multivar_df = load_data(['time','total load actual', 'price actual'], verbose=True)


df = load_data(col=["total load forecast","total load actual"])

#fill nans with linear interpolation because this is how we will fill when using the data in the models.
df_filled = df.interpolate("linear")
mm = MinMaxScaler()
df_scaled = mm.fit_transform(df_filled)

df_prep = pd.DataFrame(df_scaled, columns=df.columns)
y_true = df_prep["total load actual"]
y_pred_forecast = df_prep["total load forecast"]

### persistence 1 day
#shift series by 24 hours
# realign y_true to have the same length and time samples
y_preds_persistance_1_day = y_true.shift(24).dropna()
persistence_1_day_mae = tf.keras.losses.MAE(y_true[y_preds_persistance_1_day.index], y_preds_persistance_1_day).numpy()
persistence_1_day_mape = tf.keras.losses.MAPE(np.maximum(y_true[y_preds_persistance_1_day.index], 1e-5), np.maximum(y_preds_persistance_1_day, 1e-5)).numpy()


### persistence 3 day average
#shift by 1, 2, 3 days. Realign to have same lengths. Average days and calcualte MAE.

shift_dfs = list()
for i in range(1, 4):
    shift_dfs.append(pd.Series(y_true.shift(24 * i), name=f"d{i}"))

y_persistance_3d = pd.concat(shift_dfs, axis=1).dropna()
y_persistance_3d["avg"] = (y_persistance_3d["d1"] + y_persistance_3d["d2"] + y_persistance_3d["d3"])/3
d3_idx = y_persistance_3d.index
persistence_3day_avg_mae = tf.keras.losses.MAE(y_true[d3_idx], y_persistance_3d['avg']).numpy()
persistence_3day_avg_mape = tf.keras.losses.MAPE(np.maximum(y_true[d3_idx], 1e-5), np.maximum(y_persistance_3d['avg'], 1e-5)).numpy()


ref_error = pd.DataFrame({
    "Method": ["TSO Forecast", "Persistence 1 Day", "Persitence 3 Day Avg"],
    "MAE": [tf.keras.losses.MAE(y_true, y_pred_forecast).numpy(),
            persistence_1_day_mae,
            persistence_3day_avg_mae],
    "MAPE":[tf.keras.losses.MAPE(np.maximum(y_true, 1e-5), np.maximum(y_pred_forecast, 1e-5)).numpy(),
            persistence_1_day_mape,
            persistence_3day_avg_mape]}, 
    index=[i for i in range(3)])

print("\nSummary of Baseline Errors")
print(ref_error)
print(f"\nAverage error in MW for TSO Forecast {round(df['total load forecast'].mean()*ref_error.iloc[0,1], 2)}")


def clean_data(series):
    """Fills missing values. 
    
        Interpolate missing values with a linear approximation.
    """
    series_filled = series.interpolate(method='linear')
        
    return series_filled
        
    
def min_max_scale(dataframe):
    """ Applies MinMax Scaling
    
        Wrapper for sklearn's MinMaxScaler class.
    """
    mm = MinMaxScaler()
    return mm.fit_transform(dataframe)


def make_time_features(series):
    
    #convert series to datetimes
    times = series.apply(lambda x: x.split('+')[0])
    datetimes = pd.DatetimeIndex(times)
    
    hours = datetimes.hour.values
    day = datetimes.dayofweek.values
    months = datetimes.month.values
    
    hour = pd.Series(hours, name='hours')
    dayofw = pd.Series(day, name='dayofw')
    month = pd.Series(months, name='months')
    
    return hour, dayofw, month

hour, day, month = make_time_features(multivar_df.time)
print("Hours")
print(hour.head())
print("Day of Week")
print(day.head())
print("Months")
print(month.head())


def split_data(series, train_fraq, test_len=8760):
    """Splits input series into train, val and test.
    
        Default to 1 year of test data.
    """
    #slice the last year of data for testing 1 year has 8760 hours
    test_slice = len(series)-test_len

    test_data = series[test_slice:]
    train_val_data = series[:test_slice]

    #make train and validation from the remaining
    train_size = int(len(train_val_data) * train_fraq)
    
    train_data = train_val_data[:train_size]
    val_data = train_val_data[train_size:]
    
    return train_data, val_data, test_data


multivar_df = clean_data(multivar_df)

#add hour and month features
hours, day, months = make_time_features(multivar_df.time)
multivar_df = pd.concat([multivar_df.drop(['time'], axis=1), hours, day, months], axis=1)

#scale
multivar_df = min_max_scale(multivar_df)
train_multi, val_multi, test_multi = split_data(multivar_df, train_fraq=0.65, test_len=8760)
print("Multivarate Datasets")
print(f"Train Data Shape: {train_multi.shape}")
print(f"Val Data Shape: {val_multi.shape}")
print(f"Test Data Shape: {test_multi.shape}")
print(f"Nulls In Train {np.any(np.isnan(train_multi))}")
print(f"Nulls In Validation {np.any(np.isnan(val_multi))}")
print(f"Nulls In Test {np.any(np.isnan(test_multi))}")


def window_dataset(data, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=False, expand_dims=False):
    """ Create a windowed tensorflow dataset
    
    """

    #create a window with n steps back plus the size of the prediction length
    window = n_steps + n_horizon
    
    #expand dimensions to 3D to fit with LSTM inputs
    #creat the inital tensor dataset
    if expand_dims:
        ds = tf.expand_dims(data, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(ds)
    else:
        ds = tf.data.Dataset.from_tensor_slices(data)
    
    #create the window function shifting the data by the prediction length
    ds = ds.window(window, shift=n_horizon, drop_remainder=True)
    
    #flatten the dataset and batch into the window size
    ds = ds.flat_map(lambda x : x.batch(window))
    ds = ds.shuffle(shuffle_buffer)    
    
    #create the supervised learning problem x and y and batch
    if multi_var:
        ds = ds.map(lambda x : (x[:-n_horizon], x[-n_horizon:, :1]))
    else:
        ds = ds.map(lambda x : (x[:-n_horizon], x[-n_horizon:]))
    
    ds = ds.batch(batch_size).prefetch(1)
    
    return ds

tf.random.set_seed(42)

n_steps = 72
n_horizon = 24
batch_size = 1
shuffle_buffer = 100


ds = window_dataset(train_multi, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=True)

print('Example sample shapes')
for idx,(x,y) in enumerate(ds):
    print("x = ", x.numpy().shape)
    print("y = ", y.numpy().shape)
    break


def build_dataset(train_fraq=0.65, 
                  n_steps=24*30, 
                  n_horizon=24, 
                  batch_size=256, 
                  shuffle_buffer=500, 
                  expand_dims=False, 
                  multi_var=False):
    """If multi variate then first column is always the column from which the target is contstructed.
    """
    
    tf.random.set_seed(23)
    
    if multi_var:
        data = load_data(col=['time', 'total load actual', 'price actual'])
        hours, day, months = make_time_features(data.time)
        data = pd.concat([data.drop(['time'], axis=1), hours, day, months], axis=1)
    else:
        data = load_data(col=['total load actual'])
        
    data = clean_data(data)
    
    if multi_var:
        mm = MinMaxScaler()
        data = mm.fit_transform(data)
    
    train_data, val_data, test_data = split_data(data, train_fraq=train_fraq, test_len=8760)
    
    train_ds = window_dataset(train_data, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=multi_var, expand_dims=expand_dims)
    val_ds = window_dataset(val_data, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=multi_var, expand_dims=expand_dims)
    test_ds = window_dataset(test_data, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=multi_var, expand_dims=expand_dims)
    
    
    print(f"Prediction lookback (n_steps): {n_steps}")
    print(f"Prediction horizon (n_horizon): {n_horizon}")
    print(f"Batch Size: {batch_size}")
    print("Datasets:")
    print(train_ds.element_spec)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = build_dataset(multi_var=True)


def get_params(multivar=False):
    lr = 3e-4
    n_steps=24*30
    n_horizon=24
    if multivar:
        n_features=5
    else:
        n_features=1
        
    return n_steps, n_horizon, n_features, lr

model_configs = dict()

def cfg_model_run(model, history, test_ds):
    return {"model": model, "history" : history, "test_ds": test_ds}


def run_model(model_name, model_func, model_configs, epochs):
    
    n_steps, n_horizon, n_features, lr = get_params(multivar=True)
    train_ds, val_ds, test_ds = build_dataset(n_steps=n_steps, n_horizon=n_horizon, multi_var=True)

    model = model_func(n_steps, n_horizon, n_features, lr=lr)

    model_hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    model_configs[model_name] = cfg_model_run(model, model_hist, test_ds)
    return test_ds


def dnn_model(n_steps, n_horizon, n_features, lr):
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(n_steps, n_features)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_horizon)
    ], name='dnn')
    
    loss=tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    
    model.compile(loss=loss, optimizer='adam', metrics=['mae'])
    
    return model


dnn = dnn_model(*get_params(multivar=True))
dnn.summary()


def cnn_model(n_steps, n_horizon, n_features, lr=3e-4):
    
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=6, activation='relu', input_shape=(n_steps,n_features)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_horizon)
    ], name="CNN")
    
    loss= tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    
    model.compile(loss=loss, optimizer='adam', metrics=['mae'])
    
    return model

cnn = cnn_model(*get_params(multivar=True))
cnn.summary()


def lstm_model(n_steps, n_horizon, n_features, lr):
    
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(72, activation='relu', input_shape=(n_steps, n_features), return_sequences=True),
        tf.keras.layers.LSTM(48, activation='relu', return_sequences=False),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_horizon)
    ], name='lstm')
    
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    
    model.compile(loss=loss, optimizer='adam', metrics=['mae'])
    
    return model

lstm = lstm_model(*get_params(multivar=True))
lstm.summary()


def lstm_cnn_model(n_steps, n_horizon, n_features, lr):
    
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=6, activation='relu', input_shape=(n_steps,n_features)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(72, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(48, activation='relu', return_sequences=False),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_horizon)
    ], name="lstm_cnn")
    
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    
    model.compile(loss=loss, optimizer='adam', metrics=['mae'])
    
    return model

lstm_cnn = lstm_cnn_model(*get_params(multivar=True))
lstm_cnn.summary()


def lstm_cnn_skip_model(n_steps, n_horizon, n_features, lr):
    
    tf.keras.backend.clear_session()
    
   
    inputs = tf.keras.layers.Input(shape=(n_steps,n_features), name='main')
    
    conv1 = tf.keras.layers.Conv1D(64, kernel_size=6, activation='relu')(inputs)
    max_pool_1 = tf.keras.layers.MaxPooling1D(2)(conv1)
    conv2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(max_pool_1)
    max_pool_2 = tf.keras.layers.MaxPooling1D(2)(conv2)
    lstm_1 = tf.keras.layers.LSTM(72, activation='relu', return_sequences=True)(max_pool_2)
    lstm_2 = tf.keras.layers.LSTM(48, activation='relu', return_sequences=False)(lstm_1)
    flatten = tf.keras.layers.Flatten()(lstm_2)
    
    skip_flatten = tf.keras.layers.Flatten()(inputs)

    concat = tf.keras.layers.Concatenate(axis=-1)([flatten, skip_flatten])
    drop_1 = tf.keras.layers.Dropout(0.3)(concat)
    dense_1 = tf.keras.layers.Dense(128, activation='relu')(drop_1)
    drop_2 = tf.keras.layers.Dropout(0.3)(dense_1)
    output = tf.keras.layers.Dense(n_horizon)(drop_2)
    
    model = tf.keras.Model(inputs=inputs, outputs=output, name='lstm_skip')
    
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    
    model.compile(loss=loss, optimizer='adam', metrics=['mae'])
    
    return model

lstm_skip = lstm_cnn_skip_model(*get_params(multivar=True))
lstm_skip.summary()


tf.keras.utils.plot_model(lstm_skip, show_shapes=True)


model_configs=dict()
run_model("dnn", dnn_model, model_configs, epochs=150)
run_model("cnn", cnn_model, model_configs, epochs=150)
run_model("lstm", lstm_model, model_configs, epochs=150)
run_model("lstm_cnn", lstm_cnn_model, model_configs, epochs=150)
run_model("lstm_skip", lstm_cnn_skip_model, model_configs, epochs=150)


legend = list()

fig, axs = plt.subplots(1, 5, figsize=(25,5))

def plot_graphs(metric, val, ax, upper):
    ax.plot(val['history'].history[metric])
    ax.plot(val['history'].history[f'val_{metric}'])
    ax.set_title(key)
    ax.legend([metric, f"val_{metric}"])
    ax.set_xlabel('epochs')
    ax.set_ylabel(metric)
    ax.set_ylim([0, upper])
    
for (key, val), ax in zip(model_configs.items(), axs.flatten()):
    plot_graphs('loss', val, ax, 0.2)
print("Loss Curves")


print("MAE Curves")
fig, axs = plt.subplots(1, 5, figsize=(25,5))
for (key, val), ax in zip(model_configs.items(), axs.flatten()):
    plot_graphs('mae', val, ax, 0.6)


names = list()
performance = list()

for key, value in model_configs.items():
    names.append(key)
    mae = value['model'].evaluate(value['test_ds'])
    performance.append(mae[1])
    
performance_df = pd.DataFrame(performance, index=names, columns=['mae'])
performance_df['error_mw'] = performance_df['mae'] * df['total load forecast'].mean()
print(performance_df)    


fig, axs = plt.subplots(5, 1, figsize=(18, 10))
days = 14

vline = np.linspace(0, days*24, days+1)

for (key, val), ax in zip(model_configs.items(), axs):

    test = val['test_ds']
    preds = val['model'].predict(test)

    xbatch, ybatch = iter(test).get_next()

    ax.plot(ybatch.numpy()[:days].reshape(-1))
    ax.plot(preds[:days].reshape(-1))
    ax.set_title(key)
    ax.vlines(vline, ymin=0, ymax=1, linestyle='dotted', transform = ax.get_xaxis_transform())
    ax.legend(["Actual", "Predicted"])

plt.xlabel("Hours Cumulative")
print('First Two Weeks of Predictions')




