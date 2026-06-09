 !pip install rtdl_num_embeddings -q --no-index --find-links=/kaggle/input/jane-street-import/rtdl_num_embeddings


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
import polars as pl
from collections import OrderedDict
import sys
from tabm_reference import Model, make_parameter_groups

import warnings
warnings.filterwarnings("ignore")

import kaggle_evaluation.jane_street_inference_server

import os

import joblib

from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import Callback
import gc


feature_list = [f"feature_{idx:02d}" for idx in range(79) if idx != 61]

target_col = "responder_6" 
target_col2 = "responder_3" 

feature_test = feature_list + [f"responder_{idx}_lag_1" for idx in range(9)]\
                    + ['sin_time_id', 'cos_time_id', 'sin_time_id_halfday', 'cos_time_id_halfday', 'sin_feature_61', 'cos_feature_61']


feature_cat = ["feature_09", "feature_10", "feature_11", "symbol_id", "time_id"]
feature_cont = [item for item in feature_test if item not in feature_cat]

batch_size = 8192

std_feature = [i for i in feature_list if i not in feature_cat] + [f"responder_{idx}_lag_1" for idx in range(9)]

lag_statscols_rename = { f"responder_{idx}_lag_1" : f"responder_{idx}" for idx in range(9)}

data_stats = joblib.load("/kaggle/input/js-gbdt-tabm/data_stats.pkl")
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
        pl.col(column).map_elements(encode_category, return_dtype=pl.Int64).alias(column)
    )

class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        mse_loss = torch.sum((y_pred - y_true) ** 2)
        var_y = torch.sum(y_true ** 2)
        loss = mse_loss / (var_y + 1e-8)
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
                    'dropout': 0,
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
        X_data, y_ol, y2_ol = batch
        X_ol = X_data[:, :-2]
        symbol_ol = X_data[:, -2]
        time_ol = X_data[:, -1]

        x_cont_ol = X_ol[:, [col for col in range(X_ol.shape[1]) if col not in [9, 10, 11]]]
        x_cont_ol = x_cont_ol + torch.randn_like(x_cont_ol) * 0.02

        x_cat_ol = X_ol[:, [9, 10, 11]]
        x_cat_ol = (torch.concat([x_cat_ol, symbol_ol.unsqueeze(-1), time_ol.unsqueeze(-1)], axis=1)).to(torch.int64)

        y_hat = self(x_cont_ol, x_cat_ol)


        loss1 = self.loss_fn(y_hat[:, :, 0].flatten(0, 1), y_ol.repeat_interleave(self.k))
        loss2 = self.loss_fn(y_hat[:, :, 1].flatten(0, 1), y2_ol.repeat_interleave(self.k))
        loss = 0.85 * loss1 + 0.15 * loss2

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=x_cont_ol.size(0))

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(make_parameter_groups(self.model), lr=self.lr, weight_decay=self.weight_decay, eps=1e-4)
        return {
            'optimizer': optimizer,

        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted_metrics}")
        

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# cat_cardinalities = [23, 10, 32, 40, 969]
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# pl_model = Model(
#         n_num_features=len(feature_cont),
#         cat_cardinalities=cat_cardinalities,
#         n_classes=2,
#         backbone={
#             'type': 'MLP',
#             'n_blocks': 3 ,
#             'd_block': 512,
#             'dropout': 0,
#         },
#         bins=None,
#         num_embeddings= None,
#         # cat_embeddings={
#         #     None if cat_cardinalities is None else
#         #         'type': 'TrainablePositionEncoding',
#         #         'd_embedding' : [32, 32, 32, 32, 64],
#         #         'cardinality' : cat_cardinalities,
#         # },
#         # cat_dmodel = [32, 32, 32, 32, 64],
#         arch_type='tabm',
#         k=16,
#     )

# pl_model.load_state_dict(torch.load("/kaggle/input/js-gbdt-tabm/tabmv2_ol.model"))



# valid_original = pl.scan_parquet("/kaggle/input/jane-street-data-preprocessing/validation.parquet").sort(['date_id', 'time_id', 'symbol_id'])         
# for col in feature_cat:
#     valid_original = encode_column(valid_original, col, category_mappings[col])
 
# valid_data = valid_original \
#              .select(feature_test + [target_col, 'weight', 'symbol_id', 'time_id'])

# valid_data_tensor = torch.tensor(valid_data.collect().to_numpy(), dtype=torch.float32)
# valid_ds = TensorDataset(valid_data_tensor)
# valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)
# del valid_data_tensor
# gc.collect()

# print("Load done")

# def r2_val(y_true, y_pred, sample_weight):
#     residuals = sample_weight * (y_true - y_pred) ** 2
#     weighted_residual_sum = np.sum(residuals)

#     # Calculate weighted sum of squared true values (denominator)
#     weighted_true_sum = np.sum(sample_weight * (y_true) ** 2)

#     # Calculate weighted R2
#     r2 = 1 - weighted_residual_sum / weighted_true_sum

#     return r2

# pl_model.eval()
# valid_pred_list = []
# for valid_tensor in tqdm(valid_dl):
#     X_valid = valid_tensor[0][:, :-4].to(device)
#     y_valid = valid_tensor[0][:, -4].to(device)
#     w_valid = valid_tensor[0][:, -3].to(device)
#     symbol_valid = valid_tensor[0][:, -2].to(device)
#     time_valid = valid_tensor[0][:, -1].to(device)

    
#     x_cont_valid = X_valid[:, [col for col in range(X_valid.shape[1]) if col not in [9, 10, 11]]]
#     # x_cont_valid = x_cont_valid + torch.randn_like(x_cont_valid) * 0.02
    
#     x_cat_valid = X_valid[:, [9, 10, 11]]
#     x_cat_valid = (torch.concat([x_cat_valid, symbol_valid.unsqueeze(-1),time_valid.unsqueeze(-1)], axis=1)).to(torch.int64)

#     with torch.no_grad():
#         y_pred = pl_model(x_cont_valid, x_cat_valid)

    
#     valid_pred_list.append((y_pred[:, :, 0].mean(1), y_valid, w_valid))
# weights_eval = torch.cat([x[2] for x in valid_pred_list]).cpu().numpy()
# y_eval = torch.cat([x[1] for x in valid_pred_list]).cpu().numpy()
# prob_eval = torch.cat([x[0] for x in valid_pred_list]).cpu().numpy()
# val_r2 = r2_val(y_eval, prob_eval, weights_eval)

# print(val_r2)


previous_test: pl.DataFrame | None = None
lags_: pl.DataFrame | None = None
pre_train: pl.DataFrame | None = None
day_counts = 0

# pl_model = NN.load_from_checkpoint('/kaggle/input/js-gbdt-tabm/tabmv2_epoch05.ckpt')
pl_model = NN.load_from_checkpoint('/kaggle/input/js-gbdt-tabm/tabmv2_original.ckpt').model
# 清理GPU内存
torch.cuda.empty_cache()


print('2 gpus')
pl_model = nn.DataParallel(pl_model, device_ids=[0, 1])
pl_model.cuda(device=0)

def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction with incremental learning using the entire dataset."""
    global lags_
    global previous_test
    global pre_train
    global pl_model
    global day_counts

    # Update lags if provided
    if lags is not None:
        lags = lags.with_columns(pl.col('time_id').cast(pl.Int64))
        lags = lags.with_columns(pl.col('symbol_id').cast(pl.Int64))
        lags = lags.with_columns(
            date_id = pl.col('date_id') + 1
        )
        lags_ = lags


    # Encode categorical features
    for col in feature_cat:
        test = encode_column(test, col, category_mappings[col])
        if (lags is not None) and (col == 'symbol_id' or col == 'time_id'):
            lags = encode_column(lags, col, category_mappings[col])

    # Add time-based features
    test = test.with_columns([
        (2 * np.pi * pl.col('time_id') / 967).sin().alias('sin_time_id'),
        (2 * np.pi * pl.col('time_id') / 967).cos().alias('cos_time_id'),
        (2 * np.pi * pl.col('time_id') / 483).sin().alias('sin_time_id_halfday'),
        (2 * np.pi * pl.col('time_id') / 483).cos().alias('cos_time_id_halfday'),
        (2 * np.pi * pl.col('feature_61') / 20).sin().alias('sin_feature_61'),
        (2 * np.pi * pl.col('feature_61') / 20).cos().alias('cos_feature_61'),
    ])

    # 在开始时清理 GPU 缓存
    torch.cuda.empty_cache()

    # Incremental Learning Part
    if previous_test is not None and lags is not None:
        # Join previous test data with lags for training
        train = previous_test.join(
            lags_.select(["time_id", "symbol_id", pl.col("responder_6_lag_1").alias("responder_6"), pl.col("responder_3_lag_1").alias("responder_3")]),
            on=["time_id", "symbol_id"],
            how="left"
        )
        train = train.fill_null(0)

        # Manage training data size
        if pre_train is not None and len(pre_train) > 10000:
            pre_train = pre_train.sample(n=10000, seed=2025)

        
        if pre_train is None:
            pre_train = train
        else:
            pre_train = pl.concat([pre_train, train])
        # 检查并保留最后五万个样本
        if len(pre_train) > 50000:
            pre_train = pre_train.tail(50000)
        
        print(len(pre_train))
        # print(pre_train)

        # Prepare training data
        train_data = pre_train
        
        X_train = train_data[feature_test].to_numpy()
        y_train = pre_train['responder_6'].to_numpy()
        y2_train = pre_train['responder_3'].to_numpy()
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        symbol_ids_train = train_data.select('symbol_id').to_numpy()[:, 0]
        time_ids_train = train_data.select('time_id').to_numpy()[:, 0]
        
        X_cat_train = X_train_tensor[:, [9, 10, 11]]
        X_cont_train = X_train_tensor[:, [i for i in range(X_train_tensor.shape[1]) if i not in [9, 10, 11]]]
        
        symbol_tensor_train = torch.tensor(symbol_ids_train, dtype=torch.float32).to(device)
        time_tensor_train = torch.tensor(time_ids_train, dtype=torch.float32).to(device)
        X_cat_train = torch.concat([X_cat_train, symbol_tensor_train.unsqueeze(-1), 
                                  time_tensor_train.unsqueeze(-1)], axis=1).to(torch.int64)

        y_train_tensor = torch.FloatTensor(y_train).to(device)
        y2_train_tensor = torch.FloatTensor(y2_train).to(device)


        print()
        # 训练循环
        pl_model.train()
        optimizer = torch.optim.AdamW(make_parameter_groups(pl_model), lr=1e-4, weight_decay=8e-4)
        # criterion = nn.MSELoss()
        criterion = R2Loss()
        
        for epoch in range(5):
            outputs = pl_model(X_cont_train, X_cat_train)
            loss1 = criterion(outputs[:, :, 0].flatten(0, 1), y_train_tensor.repeat_interleave(16))
            loss2 = criterion(outputs[:, :, 1].flatten(0, 1), y2_train_tensor.repeat_interleave(16))
            loss = 0.85 * loss1 + 0.15 * loss2
            # outputs = pl_model(X_cont_train, X_cat_train)
            # loss = criterion(outputs[:, :, 0].flatten(0, 1), y_train_tensor.repeat_interleave(16))
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch + 1}/3, Loss: {loss.item()}")
        
        previous_test = pl.DataFrame()

    # Prepare test data for prediction
    time_id = test.select("time_id").to_numpy()[0]
    symbol_ids = test.select('symbol_id').to_numpy()[:, 0]
    timie_id_array = test.select("time_id").to_numpy()[:, 0]

    if not lags_ is None:

        # lags_feature = lags_.filter(pl.col("time_id") == time_id)
        # lags_feature = lags_feature.drop('time_id')
        # test = test.join(lags_feature, on=["date_id", "symbol_id"], how="left") # 用来预测

        # 下面是debug的时候用
        lags_feature = lags_.group_by(["date_id", "symbol_id"], maintain_order=True).last()
        lags_feature = lags_feature.drop('time_id')
        test = test.join(lags_feature, on=["date_id", "symbol_id"], how="left")


    else:
        test = test.with_columns(
            (pl.lit(0.0).alias(f'responder_{idx}_lag_1') for idx in range(9))
        )

    # test_ol = test_ol.with_columns([
    #     pl.col(col).fill_null(0) for col in feature_test + ['symbol_id', 'time_id']
    # ])
    
    test = test.with_columns([
        pl.col(col).fill_null(0) for col in feature_test + ['symbol_id', 'time_id']
    ])
    
    test = standardize(test, std_feature, means, stds)
    # test_ol = standardize(test_ol, std_feature, means, stds)
    
    # Prepare tensors for prediction
    X_test = test[feature_test].to_numpy()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    symbol_tensor = torch.tensor(symbol_ids, dtype=torch.float32).to(device)
    time_tensor = torch.tensor(timie_id_array, dtype=torch.float32).to(device)
    X_cat = X_test_tensor[:, [9, 10, 11]]
    X_cont = X_test_tensor[:, [i for i in range(X_test_tensor.shape[1]) if i not in [9, 10, 11]]]
    
    X_cat = torch.concat([X_cat, symbol_tensor.unsqueeze(-1), time_tensor.unsqueeze(-1)], 
                        axis=1).to(torch.int64)

    # Make predictions
    pl_model.eval()
    with torch.no_grad():
        outputs = pl_model(X_cont, X_cat)  # shape: [batch_size, 16, 2]
        outputs = outputs[:, :, 0]  # 取第一个预测值
        preds = outputs.mean(dim=1).cpu().numpy()  # 在ensemble维度上取平均

    # Store current test data for next iteration
    if previous_test is None:
        previous_test = test
    else:
        previous_test = pl.concat([previous_test, test])


    # Format predictions
    try:
        predictions = test.select('row_id').with_columns(
            pl.Series(
                name='responder_6',
                values=np.clip(preds, a_min=-5, a_max=5),
                dtype=pl.Float64,
            )
        )
    except:
        print(preds.shape)

    # Validate output
    if isinstance(predictions, pl.DataFrame):
        assert predictions.columns == ['row_id', 'responder_6']
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == ['row_id', 'responder_6']).all()
    else:
        raise TypeError('The predict function must return a DataFrame')
    assert len(predictions) == len(test)

    return predictions


%%time

EVAL =  True
if EVAL:
    test_dir = '/kaggle/input/k/i2nfinit3y/janestreet-updated-simulator-for-time-series-api/debug/test.parquet'
    lags_dir = '/kaggle/input/k/i2nfinit3y/janestreet-updated-simulator-for-time-series-api/debug/lags.parquet'
    # test_dir = '/kaggle/input/js24-rmf-submission-api-debug-with-synthetic-test/synthetic_test.parquet'
    # lags_dir = '/kaggle/input/js24-rmf-submission-api-debug-with-synthetic-test/synthetic_lag.parquet'
else:
    test_dir = '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet'
    lags_dir = '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet'

inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            test_dir,
            lags_dir
        )
    )


def weighted_zero_mean_r2(y_true, y_pred, weights):
    """
    Calculate the sample weighted zero-mean R-squared score.

    Parameters:
    y_true (numpy.ndarray): Ground-truth values for responder_6.
    y_pred (numpy.ndarray): Predicted values for responder_6.
    weights (numpy.ndarray): Sample weight vector.

    Returns:
    float: The weighted zero-mean R-squared score.
    """
    numerator = np.sum(weights * (y_true - y_pred)**2)
    denominator = np.sum(weights * y_true**2)
    
    r2_score = 1 - numerator / denominator
    return r2_score


if EVAL:

    submission_file = pd.read_parquet('/kaggle/working/submission.parquet')
    y_pred = submission_file['responder_6']
    
    valid_df = pl.read_parquet("/kaggle/input/k/i2nfinit3y/janestreet-updated-simulator-for-time-series-api/valid_df.parquet")
    
    y_true = valid_df.select("responder_6").to_numpy().reshape(-1)
    
    weights = valid_df.select("weight").to_numpy().reshape(-1)
    
    print(weighted_zero_mean_r2(y_true, y_pred, weights))

