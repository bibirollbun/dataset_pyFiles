import polars as pl
!pip install rtdl_num_embeddings -q --no-index --find-links=/kaggle/input/testnew/rtdl_num_embeddings


import os, sys, gc
import enum
import datetime
import pickle
import dill
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule
import kaggle_evaluation.jane_street_inference_server
from tanm_reference import Model, make_parameter_groups

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None


@enum.unique
class DataEnum(enum.IntEnum):
    Train = 0
    Valid = 1
    Test = 2
    Infer = 3

is_debug = False
is_rerun = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', "") != "" 
is_local = os.environ.get("DOCKER_USING", "") == "LOCAL"
num_workers = 4

if is_rerun:
    is_debug = False

def load_from_dill(model_name, model_path=None, file_ext='.dill'):
    model_object = None
    with open(f"{model_path}/{model_name}{file_ext}", "rb") as file_handle:
        model_object = dill.load(file_handle)
    return model_object


model_path = '/kaggle/input/js-2024' + ('/' if is_local else '-') + '19-02-1/last_tabm.pt'
stats_path = '/kaggle/input/js-2024' + ('/' if is_local else '-') + '19-02-1/data_stats.dill'
device = torch.device('cuda:0') # if torch.cuda.is_available() else "cpu"

target_col = "responder_6"
necessary_cols = [target_col, 'weight']
feat_clear_categ = ["feature_09", "feature_10", "feature_11"]
feature_categ = feat_clear_categ + ['symbol_id', 'time_id']
feature_cols = [f"feature_{idx:02d}" for idx in range(79) if idx not in [9, 10, 11, 61]]
responder_cols = [f"responder_{idx}_lag_1" for idx in range(9)] 
feature_cont = feature_cols + responder_cols
dataset_cols = feature_cont + necessary_cols + feature_categ
std_feature = [i for i in feature_cont]

batch_size = 8192
n_cont_features = len(feature_cont)
n_cat_features = len(feature_categ)
n_classes = None
cat_cardinalities = [23, 10, 32, 40, 969]
# TabM
arch_type = 'tabm'
bins = None
model_koef = 32

print(n_cont_features, n_cat_features, len(dataset_cols))

category_mappings = {
    'feature_09': {2: 0, 4: 1, 9: 2, 11: 3, 12: 4, 14: 5, 15: 6, 25: 7, 26: 8, 30: 9, 
        34: 10, 42: 11, 44: 12, 46: 13, 49: 14, 50: 15, 57: 16, 64: 17, 68: 18, 70: 19, 81: 20, 82: 21},
    'feature_10': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 12: 8},
    'feature_11': {9: 0, 11: 1, 13: 2, 16: 3, 24: 4, 25: 5, 34: 6, 40: 7, 48: 8, 50: 9, 59: 10, 62: 11, 63: 12, 66: 13,
        76: 14, 150: 15, 158: 16, 159: 17, 171: 18, 195: 19, 214: 20, 230: 21, 261: 22, 297: 23, 336: 24, 376: 25, 388: 26, 410: 27, 522: 28, 534: 29, 539: 30},
    'symbol_id': {i : i for i in range(39)},
    'time_id' : {i : i for i in range(968)}
}

def standardize(df, feature_cols, means, stds):
    return df.with_columns([
        ((pl.col(col) - means[col]) / stds[col]).alias(col) for col in feature_cols
    ])

def encode_column(df, column, mapping):
    max_value = max(mapping.values())
    def encode_category(category):
        return mapping.get(category, max_value + 1)
    return df.with_columns(pl.col(column).map_elements(encode_category, return_dtype=pl.Int64).alias(column))

model_tabm = Model(
    n_num_features=n_cont_features,
    cat_cardinalities=cat_cardinalities,
    n_classes=n_classes,
    backbone={
        'type': 'MLP',
        'n_blocks': 3 ,
        'd_block': 512,
        'dropout': 0.25,
    },
    bins=bins,
    num_embeddings=(
        None
        # {
        #     'type': 'PeriodicEmbeddings',
        #     'd_embedding': 16,
        #     'lite':True,
        # }
    ),
    arch_type=arch_type,
    k=model_koef,
).to(device)

with open(stats_path, "rb") as file_handle:
    data_stats = dill.load(file_handle)
means, stds = data_stats['means'], data_stats['stds']

checkpoint = torch.load(model_path, weights_only=True)
model_tabm.load_state_dict(checkpoint['model_state_dict'])
model_tabm.to(device)

lags_history = None

def predict_tabm(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    global lags_history
    for col in feature_categ:
        test = encode_column(test, col, category_mappings[col])
    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    time_id = test.select("time_id").to_numpy()[0]
    if time_id == 0:
        lags = lags.with_columns(pl.col('time_id').cast(pl.Int64))
        lags = lags.with_columns(pl.col('symbol_id').cast(pl.Int64))    
        lags_history = lags
    lags = lags_history.clone().group_by(["date_id", "symbol_id"], maintain_order=True).last() # pick up last record of previous date
    test = test.join(lags, on=["time_id", "symbol_id"], how="left")
    
    test = test.select(pl.all().forward_fill())
    test = test.with_columns([
        pl.col(col).fill_null(0) for col in feature_cont + feat_clear_categ
    ])
    test = standardize(test, std_feature, means, stds)

    model_tabm.eval()

    with torch.no_grad():
        X_cont = torch.tensor(test[feature_cont].to_numpy(), dtype=torch.float32).to(device)
        X_categ = torch.tensor(test[feature_categ].to_numpy(), dtype=torch.int64).to(device)
        outputs = model_tabm(X_cont, X_categ)
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
    
    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame)
    # with columns 'row_id', 'responer_6'
    assert list(predictions.columns) == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)
    return predictions


is_first = False

def predict(test:pl.DataFrame, lags:pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    global is_first

    pd_tabm = predict_tabm(test,lags).to_pandas()
    pd_tabm = pd_tabm.rename(columns={'responder_6': 'col_tabm'})

    if not is_first:
        display(pd_tabm)
        is_first = True

    predictions = test.select('row_id', pl.lit(0.0).alias('responder_6'))
    pred = pd_tabm['col_tabm'].to_numpy()
    predictions = predictions.with_columns(pl.Series('responder_6', pred.ravel()))
    return predictions

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

