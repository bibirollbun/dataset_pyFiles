!pip install rtdl_num_embeddings -q --no-index --find-links=/kaggle/input/jane-street-import/rtdl_num_embeddings


import os, sys, gc
import pickle
import dill
import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)

from sklearn.metrics import r2_score

import torch.optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm
from collections import OrderedDict
from tabm_reference import Model, make_parameter_groups

import warnings
import joblib
from pytorch_lightning.callbacks import Callback
import gc

import lightgbm as lgb
from lightgbm import LGBMRegressor, Booster
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

sys.path.append("/kaggle/input/jane-street-real-time-market-data-forecasting")


with open('/kaggle/input/als-e-106-pp0-40-xgb-5fold/result0.pkl', "rb") as fp:
    result = pickle.load(fp)
    print(result)
    xgb_model = result["model"]


class CONFIG:
    """Configuration class for model parameters"""
    seed = 42  # Random seed for reproducibility
    target_col = "responder_6"  # Target variable name
    # Features: 79 base features + 9 lagged features
    feature_cols = [f"feature_{idx:02d}" for idx in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
    # Paths to pre-trained models
    model_paths = [
        "/kaggle/input/js-xs-nn-trained-model",  # Neural Network models
        "/kaggle/input/js-with-lags-trained-xgb/result.pkl", # XGBoost model
        "/kaggle/input/als-e-106-pp0-40-xgb-5fold/result0.pkl",
        "/kaggle/input/als-e-106-pp0-40-xgb-5fold/result1.pkl",
        "/kaggle/input/als-e-106-pp0-40-xgb-5fold/result2.pkl",
        "/kaggle/input/als-e-106-pp0-40-xgb-5fold/result3.pkl",
        "/kaggle/input/als-e-106-pp0-40-xgb-5fold/result4.pkl"  
    ]

# Load validation data
valid = pl.scan_parquet(f"/kaggle/input/js24-preprocessing-create-lags/validation.parquet/").collect().to_pandas()

# Load XGBoost model
xgb_model = None
with open(CONFIG.model_paths[2], "rb") as fp:
    result = pickle.load(fp)
    xgb_model = result["model"]
xgb_feature_cols = ["symbol_id", "time_id"] + CONFIG.feature_cols

xgb_model2 = None
with open(CONFIG.model_paths[4], "rb") as fp:
    result = pickle.load(fp)
    xgb_model2 = result["model"]

xgb_model4 = None
with open(CONFIG.model_paths[6], "rb") as fp:
    result = pickle.load(fp)
    xgb_model4 = result["model"]

xgb_model5 = None
with open(CONFIG.model_paths[1], "rb") as fp:
    result = pickle.load(fp)
    xgb_model5 = result["model"]

def r2_val(y_true, y_pred, sample_weight):
    """
    Calculate weighted R² score
    Args:
        y_true: True values
        y_pred: Predicted values
        sample_weight: Weights for each sample
    Returns:
        Weighted R² score
    """
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2

class NN(LightningModule):
    """Neural Network model using PyTorch Lightning"""
    
    def __init__(self, input_dim, hidden_dims, dropouts, lr, weight_decay):
        """
        Initialize the neural network
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropouts: List of dropout rates
            lr: Learning rate
            weight_decay: Weight decay for regularization
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Build network architecture
        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.BatchNorm1d(in_dim))  # Batch normalization
            if i > 0:
                layers.append(nn.SiLU())  # SiLU activation (except first layer)
            if i < len(dropouts):
                layers.append(nn.Dropout(dropouts[i]))  # Dropout for regularization
            layers.append(nn.Linear(in_dim, hidden_dim))  # Linear layer
            in_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())  # Tanh activation for bounded output
        
        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_step_outputs = []

    def forward(self, x):
        """Forward pass with scaling"""
        return 5 * self.model(x).squeeze(-1)  # Scale output to [-5, 5] range

    def training_step(self, batch):
        """Single training step"""
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none') * w  # Weighted MSE loss
        loss = loss.mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch):
        """Single validation step"""
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none') * w
        loss = loss.mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.validation_step_outputs.append((y_hat, y, w))
        return loss

    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end"""
        if not self.trainer.sanity_checking:
            y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
            weights = torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()
            val_r_square = r2_val(y, prob, weights)
            self.log("val_r_square", val_r_square, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def on_train_epoch_end(self):
        """Log metrics at end of training epoch"""
        if not self.trainer.sanity_checking:
            epoch = self.trainer.current_epoch
            metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                      for k, v in self.trainer.logged_metrics.items()}
            formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
            print(f"Epoch {epoch}: {formatted_metrics}")

# Load ensemble of models (5-fold cross-validation)
N_folds = 5
models = []
for fold in range(N_folds):
    checkpoint_path = f"{CONFIG.model_paths[0]}/nn_{fold}.model"
    model = NN.load_from_checkpoint(checkpoint_path)
    models.append(model.to("cuda:0"))


# Clear validation data from memory to free up space
del valid
gc.collect()

# Global variable to store lagged features
lags_: pl.DataFrame | None = None

def predict_nn_xgb(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """
    Make predictions using ensemble of XGBoost and Neural Network models
    
    Args:
        test: DataFrame containing test data
        lags: DataFrame containing lagged features (optional)
        
    Returns:
        DataFrame with predictions
    """
    global lags_
    
    # Store lags in global variable if provided
    if lags is not None:
        lags_ = lags

    # Initialize predictions DataFrame with row_id and placeholder predictions
    predictions_nn = test.select('row_id', pl.lit(0.0).alias('responder_6',))

    # Process lagged features
    # Get last record for each date_id and symbol_id combination
    lags = lags_.clone().group_by(["date_id", "symbol_id"], maintain_order=True).last()
    
    # Join test data with lagged features
    test = test.join(lags, on=["date_id", "symbol_id"], how="left")

    # Initialize arrays for model predictions
    preds_xgb = np.zeros((test.shape[0],))  # XGBoost predictions
    preds_nn = np.zeros((test.shape[0],))   # Neural Network predictions

    # Generate XGBoost predictions
    preds_xgb += xgb_model.predict(test[xgb_feature_cols].to_pandas()) * 0.25
    preds_xgb += xgb_model2.predict(test[xgb_feature_cols].to_pandas()) * 0.25
    preds_xgb += xgb_model4.predict(test[xgb_feature_cols].to_pandas()) * 0.25
    preds_xgb += xgb_model5.predict(test[xgb_feature_cols].to_pandas()) * 0.25

    # Generate Neural Network predictions
    # Prepare input data
    test_input = test[CONFIG.feature_cols].to_pandas()
    # Handle missing values: forward fill then fill remaining with zeros
    test_input = test_input.fillna(method='ffill').fillna(0)
    # Convert to PyTorch tensor and move to GPU
    test_input = torch.FloatTensor(test_input.values).to("cuda:0")

    # Generate predictions from Neural Network ensemble
    with torch.no_grad():  # Disable gradient calculation for inference
        for i, nn_model in enumerate(models):
            nn_model.eval()  # Set model to evaluation mode
            # Average predictions from all models
            preds_nn += nn_model(test_input).cpu().numpy() / len(models)

    # Combine predictions with equal weights (50% XGBoost, 50% Neural Network)
    preds = 0.55 * preds_xgb + 0.45 * preds_nn

    # Create final predictions DataFrame
    predictions_nn = test.select('row_id').\
        with_columns(
            pl.Series(
                name='responder_6',
                values=np.clip(preds, a_min=-5, a_max=5),  # Clip predictions to [-5, 5] range
                dtype=pl.Float64,
            )
        )

    return predictions_nn


feature_list = [f"feature_{idx:02d}" for idx in range(79) if idx != 61]

target_col = "responder_6" 

feature_test = feature_list \
                + [f"responder_{idx}_lag_1" for idx in range(9)] 

feature_cat = ["feature_09", "feature_10", "feature_11"]
feature_cont = [item for item in feature_test if item not in feature_cat]

batch_size = 8192

std_feature = [i for i in feature_list if i not in feature_cat] + [f"responder_{idx}_lag_1" for idx in range(9)]

data_stats = joblib.load("/kaggle/input/my-own-js/data_stats.pkl")
means = data_stats['mean']
stds = data_stats['std']

def standardize(df, feature_cols, means, stds):
    return df.with_columns([
        ((pl.col(col) - means[col]) / stds[col]).alias(col) for col in feature_cols
    ])

category_mappings = {'feature_09': {2: 0, 4: 1, 9: 2, 11: 3, 12: 4, 14: 5, 15: 6, 25: 7, 26: 8, 30: 9, 34: 10, 42: 11, 44: 12, 46: 13, 49: 14, 50: 15, 57: 16, 64: 17, 68: 18, 70: 19, 81: 20, 82: 21},
 'feature_10': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 12: 8},
 'feature_11': {9: 0, 11: 1, 13: 2, 16: 3, 24: 4, 25: 5, 34: 6, 40: 7, 48: 8, 50: 9, 59: 10, 62: 11, 63: 12, 66: 13,
  76: 14, 150: 15, 158: 16, 159: 17, 171: 18, 195: 19, 214: 20, 230: 21, 261: 22, 297: 23, 336: 24, 376: 25, 388: 26, 410: 27, 522: 28, 534: 29, 539: 30},
 'symbol_id': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19,
  20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38},
 'time_id' : {i : i for i in range(968)}}

def encode_column(df, column, mapping):
    max_value = max(mapping.values())  

    def encode_category(category):
        return mapping.get(category, max_value + 1)  
    
    return df.with_columns(
        pl.col(column).map_elements(encode_category).alias(column)
    )

class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        mse_loss = torch.sum((y_pred - y_true) ** 2)
        var_y = torch.sum(y_true ** 2)
        loss = mse_loss / (var_y + 1e-38)
        return loss

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
                    'd_block': 512,
                    'dropout': 0.25,
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
        self.loss_fn = R2Loss()
        # self.loss_fn = weighted_mse_loss

    def forward(self, x_cont, x_cat):
        return self.model(x_cont, x_cat).squeeze(-1)

    def training_step(self, batch):
        x_cont,x_cat, y, w , w_y= batch
        x_cont = x_cont + torch.randn_like(x_cont) * 0.02
        y_hat = self(x_cont, x_cat)
        # loss = self.loss_fn(y_hat.flatten(0, 1), y.repeat_interleave(self.k), w_y.repeat_interleave(self.k))
        loss = self.loss_fn(y_hat.flatten(0, 1), y.repeat_interleave(self.k))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=x_cont.size(0))
        self.training_step_outputs.append((y_hat.mean(1), y, w))
        return loss

    def validation_step(self, batch):
        x_cont,x_cat, y, w, w_y = batch
        x_cont = x_cont + torch.randn_like(x_cont) * 0.02
        y_hat = self(x_cont, x_cat)
        # loss = self.loss_fn(y_hat.flatten(0, 1), y.repeat_interleave(self.k), w_y.repeat_interleave(self.k))
        loss = self.loss_fn(y_hat.flatten(0, 1), y.repeat_interleave(self.k))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=x_cont.size(0))
        self.validation_step_outputs.append((y_hat.mean(1), y, w))
        return loss

    def on_validation_epoch_end(self):
        """Calculate validation WRMSE at the end of the epoch."""
        y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
        if self.trainer.sanity_checking:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
        else:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
            weights = torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()
            # r2_val
            val_r_square = r2_val(y, prob, weights)
            self.log("val_r_square", val_r_square, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(make_parameter_groups(self.model), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5,
        #                                                        verbose=True)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'monitor': 'val_r_square',
            # }
        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        y = torch.cat([x[1] for x in self.training_step_outputs]).cpu().numpy()
        prob = torch.cat([x[0] for x in self.training_step_outputs]).detach().cpu().numpy()
        weights = torch.cat([x[2] for x in self.training_step_outputs]).cpu().numpy()
        # r2_training
        train_r_square = r2_val(y, prob, weights)
        self.log("train_r_square", train_r_square, prog_bar=True, on_step=False, on_epoch=True)
        self.training_step_outputs.clear()

        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted_metrics}")
        
class custom_args():
    def __init__(self):
        self.usegpu = True
        self.gpuid = 0
        self.seed = 42
        self.model = 'nn'
        self.use_wandb = False
        self.project = 'js-tabm-with-lags'
        self.dname = "./input_df/"
        self.loader_workers = 10   
        self.bs = 8192
        self.lr = 1e-3
        self.weight_decay = 8e-4
        self.n_cont_features = 84
        self.n_cat_features = 5
        self.n_classes = None
        self.cat_cardinalities = [23, 10, 32, 40, 969]
        self.patience = 7
        self.max_epochs = 10
        self.N_fold = 5


my_args = custom_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = NN.load_from_checkpoint('/kaggle/input/my-own-js/tabm_epochepoch03.ckpt').to(device)


lags_ : pl.DataFrame | None = None

lags_history = None

def predict_tabm(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    global lags_, lags_history
    if lags is not None:
        lags_ = lags
    
    for col in feature_cat + ['symbol_id', 'time_id']:
        test = encode_column(test, col, category_mappings[col])

    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    
    symbol_ids = test.select('symbol_id').to_numpy()[:, 0]

    time_id = test.select("time_id").to_numpy()[0]
    timie_id_array = test.select("time_id").to_numpy()[:, 0]
    
    if time_id == 0:
        lags = lags.with_columns(pl.col('time_id').cast(pl.Int64))
        lags = lags.with_columns(pl.col('symbol_id').cast(pl.Int64))
    
        lags_history = lags
        lags = lags.filter(pl.col("time_id") == 0)
        
        
        test = test.join(lags, on=["time_id", "symbol_id"],  how="left")
    else:
        lags = lags_history.filter(pl.col("time_id") == time_id)
        test = test.join(lags, on=["time_id", "symbol_id"],  how="left")

    
    test = test.with_columns([
        pl.col(col).fill_null(0) for col in feature_list + [f"responder_{idx}_lag_1" for idx in range(9)] 
    ])

    test = standardize(test, std_feature, means, stds)
    
    X_test = test[feature_test].to_numpy()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    symbol_tensor = torch.tensor(symbol_ids, dtype=torch.float32).to(device)
    time_tensor = torch.tensor(timie_id_array, dtype=torch.float32).to(device)
    X_cat = X_test_tensor[:, [9, 10, 11]]
    X_cont = X_test_tensor[:, [i for i in range(X_test_tensor.shape[1]) if i not in [9, 10, 11]]]
    # X_cont = X_cont + torch.randn_like(X_cont) * 0.02

    X_cat = (torch.concat([X_cat, symbol_tensor.unsqueeze(-1), time_tensor.unsqueeze(-1)], axis=1)).to(torch.int64)
    
    model.eval()
    with torch.no_grad():
        
        outputs = model(X_cont, X_cat)
        # Assuming the model outputs a tensor of shape (batch_size, 1)
        preds = outputs.squeeze(-1).cpu().numpy()
        preds = preds.mean(1)
    
    predictions = \
    test.select('row_id').\
    with_columns(
        pl.Series(
            name   = 'responder_6', 
            values = np.clip(preds, a_min = -5, a_max = 5),
            dtype  = pl.Float64,
        )
    )

    return predictions    


feature_cols = [f"feature_{idx:02d}" for idx in range(79)]+ [f"responder_{idx}_lag_1" for idx in range(9)]
target_col = "responder_6"
selected_features = ["symbol_id", "time_id"] + feature_cols+[target_col]
features = ["symbol_id", "time_id"] + feature_cols


def features_engineering(df):
    df['feature_16_17_interaction'] = df['feature_16'] * df['feature_17']
    df['feature_36_squared'] = df['feature_36'] ** 2
    df['feature_16_36_interaction'] = df['feature_16'] * df['feature_36']
    df['feature_ratio'] = df['feature_16'] / (df['feature_17'] + 1e-9)
    df['feature_16_rolling_mean'] = df['feature_16'].rolling(window=5, min_periods=1).mean()
    df['feature_16_rolling_std'] = df['feature_16'].rolling(window=5, min_periods=1).std()
    return df


fs=[f"feature_{idx:02d}" for idx in range(79)]
rs=[f"responder_{idx}" for idx in range(9)]

n_features=['feature_16_17_interaction','feature_36_squared','feature_16_36_interaction','feature_ratio','feature_16_rolling_mean','feature_16_rolling_std']
f_feat=fs+['symbol_id','time_id']+n_features


lags_ : pl.DataFrame | None = None

def predict_ridge(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    global lags_
    if lags is not None:
        lags_ = lags

    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    symbol_ids = test.select('symbol_id').to_numpy()[:, 0]

    lags = lags_.clone().group_by(["date_id", "symbol_id"], maintain_order=True).last() # pick up last record of previous date
    test = test.join(lags, on=["date_id", "symbol_id"],  how="left")

    rdg = joblib.load('/kaggle/input/ridgemodel/model_ridgesy_4.pkl')

    test=features_engineering(test[features].to_pandas())
    test_preds = rdg.predict(test[f_feat].fillna(3).values)
    
    # Add predictions to result DataFrame
    predictions = predictions.with_columns(pl.Series('responder_6', test_preds.ravel()))
    return predictions


def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """
    Make ensemble predictions combining all three models:
    - Neural Network + XGBoost ensemble
    - Starter ensemble (LightGBM + CatBoost + XGBoost)
    - Ridge Regression
    
    Args:
        test: DataFrame containing test data
        lags: DataFrame containing lagged features
        
    Returns:
        DataFrame with final ensemble predictions
    """
    
    # Get predictions from each model/ensemble
    pd_nn_xgb = predict_nn_xgb(test, lags).to_pandas()     # Neural Network + XGBoost ensemble
    pd_ridge = predict_ridge(test, lags).to_pandas()  # LGB+XGB+Ridge Regression
    pd_tabm = predict_tabm(test, lags).to_pandas()  

    # Rename prediction columns to avoid conflicts
    pd_nn_xgb = pd_nn_xgb.rename(columns={'responder_6': 'col_nn_xgb'})
    pd_ridge = pd_ridge.rename(columns={'responder_6': 'col_ridge'})
    pd_tabm  = pd_tabm.rename(columns={'responder_6': 'col_tabm'})

    # Merge all predictions based on row_id
    pds = pd.merge(pd_nn_xgb, pd_ridge, on=['row_id'])
    pds = pd.merge(pds, pd_tabm, on=['row_id'])

    e_weights = [0.65, 0.25, 0.40]
    # Create final weighted ensemble prediction:
    pds['responder_6'] = (
        pds['col_nn_xgb'] * e_weights[0] +
        pds['col_ridge'] * e_weights[1] +
        pds['col_tabm'] * e_weights[2]
    )

    # Create final predictions DataFrame in required format
    predictions = test.select('row_id', pl.lit(0.0).alias('responder_6'))
    pred = pds['responder_6'].to_numpy()
    predictions = predictions.with_columns(
        pl.Series(
            name   = 'responder_6', 
            values = np.clip(pred.ravel(), a_min = -5, a_max = 5),
            dtype  = pl.Float64,
        ))    
    
    return predictions


import kaggle_evaluation.jane_street_inference_server
inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
        )
    )

print('submit')

