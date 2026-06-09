!pip install zarr -q --no-index --find-links=/kaggle/input/zarr-package/


!cp -r /kaggle/input/jane-street-market-library/data/symbol_data.zarr /kaggle/working/sysymbol_data.zarr


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl
import zarr
from zarr import ProcessSynchronizer
from zarr import DirectoryStore
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn import set_config
import torch
from tqdm import tqdm, trange
from numpy.typing import NDArray
from functools import lru_cache
from omegaconf import OmegaConf
import json
import sys
sys.path.append('/kaggle/input/jane-street-market-library')

set_config(transform_output = "default")

store = DirectoryStore('/kaggle/working/symbol_data.zarr')
synchronizer = ProcessSynchronizer('/kaggle/working/sync.lock')
root = zarr.group(store, overwrite=False, synchronizer=synchronizer)
symb_last_dates = {k:0 for k in range(39)}
latest_time_idx = 0 
with open('/kaggle/input/jane-street-market-library/data/categories.json') as f:
    categories = json.load(f)


train_data = pl.scan_parquet('/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet')


from time import perf_counter


# @torch.compile
def format_dec_data(dec_tensor: list[torch.Tensor]):
    dec_tensor_ = torch.stack(dec_tensor)
    return {
        "decoder_lengths": torch.tensor(dec_tensor_.shape[0]).to(dtype=torch.int64),
        "decoder_categoricals": dec_tensor_[:, -5:].to(dtype=torch.int64).long(),
        "decoder_reals": dec_tensor_[:, :-5].to(dtype=torch.float32),
    }


# @torch.compile
def merge_current_data(
    curr_day_dec_store: dict[int, list[torch.Tensor]], timestep: torch.Tensor
):
    for data_idx in range(timestep.size(0)):
        symbol_id = int(timestep[data_idx, -1].item())
        symbol_data = timestep[data_idx, :]
        if symbol_id not in curr_day_dec_store:
            curr_day_dec_store[symbol_id] = [symbol_data]
        else:
            curr_day_dec_store[symbol_id].append(symbol_data)
    # print({k:len(v) for k,v in curr_day_dec_store.items()})
    return curr_day_dec_store

# @torch.compile
def format_enc_data(data: torch.Tensor):
    device = data.device
    return {
        "encoder_lengths": torch.tensor(data.shape[0]).to(dtype=torch.int64, device=device),
        "encoder_categoricals": data[:, :-9][:, -5:].to(dtype=torch.int64, device=device).long(),
        "encoder_reals": data[:, :79].to(dtype=torch.float32, device=device),
        "encoder_targets": data[:, -9:].to(dtype=torch.float32, device=device),
    }

class DataStore:
    def __init__(self, *args, **kwargs):
        self.curr_day_dec_store: dict[str, list[torch.Tensor]] = {}
        self.store = DirectoryStore("/kaggle/working/symbol_data.zarr")
        self.synchronizer = ProcessSynchronizer("/kaggle/working/sync.lock")
        self.root = zarr.group(store, overwrite=False, synchronizer=synchronizer)
        self.symb_last_dates = {k: 1695 for k in range(39)}
        self.latest_time_idx = 0
        self.col_order = ['row_id', 'date_id', 'time_id', 'symbol_id', 'weight', 'feature_00', 'feature_01', 'feature_02', 'feature_03', 'feature_04', 'feature_05', 'feature_06', 'feature_07', 'feature_08', 'feature_09', 'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30', 'feature_31', 'feature_32', 'feature_33', 'feature_34', 'feature_35', 'feature_36', 'feature_37', 'feature_38', 'feature_39', 'feature_40', 'feature_41', 'feature_42', 'feature_43', 'feature_44', 'feature_45', 'feature_46', 'feature_47', 'feature_48', 'feature_49', 'feature_50', 'feature_51', 'feature_52', 'feature_53', 'feature_54', 'feature_55', 'feature_56', 'feature_57', 'feature_58', 'feature_59', 'feature_60', 'feature_61', 'feature_62', 'feature_63', 'feature_64', 'feature_65', 'feature_66', 'feature_67', 'feature_68', 'feature_69', 'feature_70', 'feature_71', 'feature_72', 'feature_73', 'feature_74', 'feature_75', 'feature_76', 'feature_77', 'feature_78', 'is_scored']
        self.categories = {
            "feature_09": [
                2,
                4,
                9,
                11,
                12,
                14,
                15,
                25,
                26,
                30,
                34,
                42,
                44,
                46,
                49,
                50,
                57,
                64,
                68,
                70,
                # 81,
                # 82,
            ],
            "feature_10": [1, 2, 3, 4, 5, 6, 7,
                           # 12
                          ],
            "feature_11": [
                9,
                11,
                13,
                16,
                24,
                25,
                34,
                40,
                48,
                50,
                59,
                62,
                63,
                66,
                76,
                150,
                158,
                159,
                171,
                195,
                214,
                230,
                261,
                297,
                336,
                376,
                388,
                410,
                522,
                # 534,
                # 539,
            ],
            "time_id": [i for i in range(968)],
            "symbol_id":[i for i in range(38)]
        }
        self.symbol_idx = 3
        self.time_id_idx = 2
        self.weight_idx = 4
        self.date_id_scaler = StandardScaler()
        self.date_id_scaler.fit(np.arange(0, 1900).reshape(-1, 1))
        self.time_idx_scaler = StandardScaler()
        self.time_idx_scaler.fit(np.arange(0, 1854468).reshape(-1, 1))
        self.real_scaler = StandardScaler()
        self.feat_idx = [
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
        ]
        self.dec_ct = ColumnTransformer(
            transformers=[
                ("real_time_indices", "passthrough", [0, 1]),
                ("weight_idx", "passthrough", [self.weight_idx]),
                ("features", StandardScaler(), self.feat_idx),
                (
                    "cat_encoder_1",
                    OrdinalEncoder(
                        categories=[
                            self.categories["feature_09"],
                        ],
                        handle_unknown="use_encoded_value",
                        unknown_value=len(categories["feature_09"])-1
                    ),
                    [14],
                ),
                (
                    "cat_encoder_2",
                    OrdinalEncoder(
                        categories=[
                            self.categories["feature_10"],
                        ],
                        handle_unknown="use_encoded_value",
                        unknown_value=len(categories["feature_10"])-1,
                    ),
                    [15],
                ),
                (
                    "cat_encoder_3",
                    OrdinalEncoder(
                        categories=[
                            self.categories["feature_11"],
                        ],
                        handle_unknown="use_encoded_value",
                        unknown_value=len(categories["feature_11"])-1,
                    ),
                    [16],
                ),
                ("static_real", "passthrough", [self.time_id_idx, self.symbol_idx]),
            ],
            remainder="drop",
        )
        self.tensor_device = "cpu"
        self.old_symb_order = []
        self.symb_order_changed = False
        self.curr_symb_order = []
        self.new_symb_order = []
        self.curr_day_enc_data = None
        self.pred_device = "cuda"
        self.is_scored = None

    def prep_dec_df(self, timestep: pl.DataFrame) -> torch.Tensor:
        symb_scaled = self.dec_ct.fit_transform(timestep.to_numpy())
        symb_scaled[:, 0] = self.time_idx_scaler.transform(
            symb_scaled[:, 0].reshape(-1, 1)
        ).ravel()
        symb_scaled[:, 1] = self.date_id_scaler.transform(
            symb_scaled[:, 1].reshape(-1, 1)
        ).ravel()
        return torch.tensor(symb_scaled, dtype=torch.float32, device=self.pred_device)  ## Changed

    def add_time_idx(self, timestep: pl.DataFrame) -> pl.DataFrame:
        curr_date = timestep["date_id"].max()
        if curr_date < 677:
            timestep = timestep.with_columns(pl.col("date_id") + 1699)
        return (
            timestep.with_columns(
                (
                    (pl.col("date_id").cast(pl.Int32) - 677) * 968
                    + pl.col("time_id").cast(pl.Int32)
                    + (677 * 849)
                ).alias("time_idx")
            )
            .fill_null(0)
            .select(
                pl.col("time_idx"),
                pl.col(
                    [
                        x
                        for x in timestep.columns
                        if x not in ["time_idx", "partition_id"]
                    ]
                ),
            )
        )

    def flush_curr_day_data(self, lags: pl.DataFrame):
        symb_lags = lags.sort("time_id").partition_by("symbol_id", maintain_order=True)
        curr_date = None
        for symb_lag in symb_lags:
            self.symb_last_dates[symb_lag["symbol_id"].min()] = symb_lag[
                "date_id"
            ].min()
            curr_symbol = symb_lag["symbol_id"].min()
            if curr_date is None:
                curr_date = symb_lag["date_id"].min()
            symb_lag = (
                symb_lag.select("^responder_.*$")
                .to_torch()
                .to(dtype=torch.float32, device=self.tensor_device)
            )
            combined_data = torch.cat(
                [torch.stack(self.curr_day_dec_store[curr_symbol],dim=0).detach().cpu(), symb_lag], dim=-1
            )
            self.root[f"{curr_symbol}/{curr_date}"] = zarr.array(
                combined_data.detach().cpu().numpy(), chunks=(968, 93), dtype=np.float32
            )
        self.curr_day_dec_store = {}
        self.curr_symb_order = []
        self.get_prev_data.cache_clear()
        self.curr_day_enc_data = None
        self.curr_batch_enc_data = None

    def update_curr_day_timestep(self, timestep: pl.DataFrame):
        # Since Everything is arriving in order no need for sorting
        # timestep = timestep.select(self.col_order)
        self.is_scored = timestep['is_scored'].to_numpy().ravel().astype('bool')
        self.is_scored  = torch.tensor(self.is_scored, device=self.pred_device, dtype=torch.bool)
        self.row_id = timestep['row_id'].to_numpy().ravel()
        timestep = timestep.drop(['row_id', 'is_scored'])
        timestep = self.add_time_idx(timestep)
        # print(timestep)
        timestep: torch.Tensor = self.prep_dec_df(timestep)
        
        curr_symb_order = timestep[:, -1].cpu().numpy().tolist()
        # print(timestep[:,-1])
        
        # curr_symb_order = curr_symb_order[is_scored]
        # curr_symb_order = curr_symb_order.tolist()
        
        if self.curr_symb_order != curr_symb_order:
            self.symb_order_changed = True
            self.new_symb_order = curr_symb_order
            self.old_symb_order = self.curr_symb_order
            self.curr_symb_order = curr_symb_order
        else:
            self.symb_order_changed = False
        self.curr_day_dec_store = merge_current_data(self.curr_day_dec_store, timestep)

    @lru_cache(maxsize=None)
    def get_prev_data(self, symbol_id: int, sampling=8):
        # print(f'Symbol ID {symbol_id}')
        data = torch.tensor(
            self.root[f"{int(symbol_id)}/{self.symb_last_dates[int(symbol_id)]}"].oindex[::sampling],
            dtype=torch.float32, device=self.pred_device
        )
        data_dict = format_enc_data(data)
        return data_dict
    
    def get_curr_batch(self):
        if self.curr_day_enc_data is None:
            self.curr_day_enc_data = {x: self.get_prev_data(x) for x in self.curr_symb_order}
            self.curr_batch_enc_data = {
                "encoder_reals": torch.stack([self.curr_day_enc_data[x]["encoder_reals"] for x in self.curr_symb_order]).to(self.pred_device),
                "encoder_lengths": torch.stack([self.curr_day_enc_data[x]["encoder_lengths"] for x in self.curr_symb_order]).to(self.pred_device),
                "encoder_targets": torch.stack([self.curr_day_enc_data[x]["encoder_targets"] for x in self.curr_symb_order]).to(self.pred_device),
                "encoder_categoricals": torch.stack([self.curr_day_enc_data[x]["encoder_categoricals"] for x in self.curr_symb_order]).to(self.pred_device),
            }
        if self.symb_order_changed:
            self.curr_batch_enc_data = {
            "encoder_reals": torch.stack([self.curr_day_enc_data[x]["encoder_reals"] for x in self.curr_symb_order]).to(self.pred_device),
            "encoder_lengths": torch.stack([self.curr_day_enc_data[x]["encoder_lengths"] for x in self.curr_symb_order]).to(self.pred_device),
            "encoder_targets": torch.stack([self.curr_day_enc_data[x]["encoder_targets"] for x in self.curr_symb_order]).to(self.pred_device),
            "encoder_categoricals": torch.stack([self.curr_day_enc_data[x]["encoder_categoricals"] for x in self.curr_symb_order]).to(self.pred_device),
            }
        dec_data = {x:format_dec_data(self.curr_day_dec_store[x]) for x in self.curr_symb_order}
        self.curr_batch_enc_data.update({
            "decoder_reals": torch.stack([dec_data[x]["decoder_reals"] for x in self.curr_symb_order]).to(self.pred_device),
            "decoder_lengths": torch.stack([dec_data[x]["decoder_lengths"] for x in self.curr_symb_order]).to(self.pred_device),
            "decoder_categoricals": torch.stack([dec_data[x]["decoder_categoricals"] for x in self.curr_symb_order]).to(self.pred_device),
        })
        return self.curr_batch_enc_data, self.is_scored


def initialise_store(train_data=train_data):
    DS = DataStore()
    last_day = train_data.filter(pl.col("date_id") == 1698).collect()
    lags = last_day.select(
        [
            "date_id",
            "time_id",
            "symbol_id",
            "^responder_.*$",
        ]
    )
    test_data_timesteps = last_day.drop('^responder_.*$').with_row_index('row_id').with_columns(pl.lit(False).alias('is_scored')).partition_by('time_id')
    print('Loading the Initial Data')
    for ts in tqdm(test_data_timesteps):
        DS.update_curr_day_timestep(ts)
        pred = np.zeros(39)
        if sum(DS.is_scored) > 0:
            p, s = DS.get_curr_batch()
            p = {k:v[s] for k, v in p.items()}
            B,_,_ = p['encoder_reals'].size()
            if B!=0:
                o = model_1(p)
                pred[s] = o.detach().cpu().numpy().ravel()
        
        # break
        
        
    DS.flush_curr_day_data(lags)

    return DS


# Import Main Config File
from jane_street.layers.tft import TFT
from jane_street.layers.embeddings import get_embedding_size
from jane_street.models.temporal_ft import TemporalFT
from jane_street.models.tsmixer import  TSMixerExtModel
# with open('/kaggle/input/jane-street-market-library/configs/master_config.yaml', 'r') as fp:
#     config = OmegaConf.load(fp)

def make_tft_config():
    with open('/kaggle/input/jane-street-market-library/data/categories.json','r') as fp:
        categories = json.load(fp)
    conf = OmegaConf.load("/kaggle/input/jane-street-market-library/configs/master_config.yaml")
    ## Data config
    # conf.limit_val_batches = 800
    # conf.val_dataset_size = 256*800*n_devices
    conf.x_categoricals = list(categories.keys())
    
    conf.hidden_size = 8
    cat_sizes = {
        k: (len(v), get_embedding_size(len(v), 16))
        for k, v in categories.items()
    }
    conf.embedding_sizes = cat_sizes
    conf.real_hidden_size = conf.hidden_size
    conf.log_interval = 2
    conf.log_val_interval = 3
    conf.lstm_layers = 2
    conf.n_heads = 4
    conf.dropout = 0.1
    conf.learning_rate = 0.0001
    conf.quantiles = [0.3, 0.5, 0.7]
    conf.weight_decay = 0.001
    conf.max_encoder_length = 968
    conf.output_size = 1
    conf.causal_attention = True
    return conf

# config = make_tft_config()
# config.x_categoricals
tft = TemporalFT.load_from_checkpoint('/kaggle/input/jane-street-market-library/main_ckpts/epoch=20-val_score=0.43-temporal-ft.ckpt')
# tft = TFT(config)
# tft = TFT(config)
tft = tft.to('cuda').eval()


tft


# Replace this function with your inference code.
# You can return either a Pandas or Polars dataframe, though Polars is recommended.
# Each batch of predictions (except the very first) must be returned within 1 minute of the batch features being provided.
started = False
lags_ = None
days_passed = 0
DS = initialise_store()
def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # All the responders from the previous day are passed in at time_id == 0. We save them in a global variable for access at every time_id.
    # Use them as extra features, if you like.
    global lags_, DS, tft, started, days_passed
    
    curr_time_id = test['time_id'].min()
    if curr_time_id == 0 and started and lags is not None:
        DS.flush_curr_day_data(lags)
        lags_ = lags
        days_passed+=1
    started=True
    # if days_passed % 30 ==0:
        
    # if lags is not None:
    #     lags_ = lags
    with torch.no_grad():
        DS.update_curr_day_timestep(test)
        # Replace this section with your own predictions
        predictions = test.select(
            'row_id',
            pl.lit(0.0).alias('responder_6'),
        )
        # pred = np.zeros_like(s)
        pred = np.zeros_like(predictions['responder_6'].to_numpy().ravel())
        # print(DS.curr_symb_order)
        if sum(DS.is_scored) > 0:
            p, s = DS.get_curr_batch()
            # print(p['decoder_categoricals'][0])
            p = {k:v[s] for k, v in p.items()}
            B,_,_ = p['encoder_reals'].size()
            if B!=0: # Only Predict if scored
                o = tft(p)
                s = s.detach().cpu().numpy().ravel()
                pred[s] = o.detach().cpu().numpy().ravel()
        predictions = predictions.with_columns(pl.Series('responder_6', pred.ravel()))
        if isinstance(predictions, pl.DataFrame):
            assert predictions.columns == ['row_id', 'responder_6']
        elif isinstance(predictions, pd.DataFrame):
            assert (predictions.columns == ['row_id', 'responder_6']).all()
        else:
            raise TypeError('The predict function must return a DataFrame')
        # Confirm has as many rows as the test data.
        assert len(predictions) == len(test)
        # print(predictions)
    return predictions


import kaggle_evaluation.jane_street_inference_server

import os

import pandas as pd
import polars as pl

test_data = pl.read_parquet('/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet')
lags =  pl.read_parquet('/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet')
# predict(test_data, lags)

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





