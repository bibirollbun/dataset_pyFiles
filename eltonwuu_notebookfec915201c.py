#env & data
import os
from pathlib import Path
import pandas as pd
import numpy as np


os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

IN_KAGGLE = Path("/kaggle/input").exists()
DATA_DIR  = Path("/kaggle/input/optiver-realized-volatility-prediction") if IN_KAGGLE else Path("data")
WORK_DIR  = Path("/kaggle/working") if IN_KAGGLE else Path(".")
WORK_DIR.mkdir(parents=True, exist_ok=True)

assert (DATA_DIR / "train.csv").exists() and (DATA_DIR / "test.csv").exists(), "需要 train.csv / test.csv"


data_dir = str(DATA_DIR) + "/"

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")
DEFAULT_PRED = float(train_df["target"].mean())  

print("Train/Test:", train_df.shape, test_df.shape, " DEFAULT_PRED:", DEFAULT_PRED)



# IO & selected columns
from typing import Optional


READ_SELECTED_COLUMNS = False

BOOK_COLS  = ["time_id","seconds_in_bucket","bid_price1","ask_price1","bid_size1","ask_size1","wap"] if READ_SELECTED_COLUMNS else None
TRADE_COLS = ["time_id","seconds_in_bucket","price","size","order_count"] if READ_SELECTED_COLUMNS else None

def stock_book_path(split: str, sid: int) -> Path:
    return DATA_DIR / f"book_{split}.parquet" / f"stock_id={sid}"

def stock_trade_path(split: str, sid: int) -> Path:
    return DATA_DIR / f"trade_{split}.parquet" / f"stock_id={sid}"

def read_parquet_safe(path: Path, columns: Optional[list] = None):
    if path.exists():
        if columns is None:
            return pd.read_parquet(path, engine="pyarrow")
        else:
            return pd.read_parquet(path, columns=columns, engine="pyarrow")
   
    return pd.DataFrame(columns=columns or [])



import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings('ignore')

def log_return(series):
    return np.log(series).diff()

def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

def generate_all_features_fixed(stock_id, data_dir='data/', data_type='train'):

    book_path = os.path.join(data_dir, f'book_{data_type}.parquet/stock_id={stock_id}')
    trade_path = os.path.join(data_dir, f'trade_{data_type}.parquet/stock_id={stock_id}')
    df_book = pd.read_parquet(book_path)
    df_trade = pd.read_parquet(trade_path)

    df_book['wap'] = (df_book['bid_price1'] * df_book['ask_size1'] + df_book['ask_price1'] * df_book['bid_size1']) / (df_book['bid_size1'] + df_book['ask_size1'])

    def process_time_id_group(group):

        group = group.drop(columns=['time_id'])
        
        group = group.set_index('seconds_in_bucket')
        full_index = pd.Index(np.arange(600), name='seconds_in_bucket')
        group = group.reindex(full_index).fillna(method='ffill')
        group['log_return'] = np.log(group['wap']).diff()
        return group

    df_book_processed = df_book.groupby('time_id').apply(process_time_id_group).reset_index()

    df_book_processed['spread'] = (df_book_processed['ask_price1'] - df_book_processed['bid_price1']) / df_book_processed['wap']
    df_book_processed['obi'] = (df_book_processed['bid_size1'] - df_book_processed['ask_size1']) / (df_book_processed['bid_size1'] + df_book_processed['ask_size1'])

    windows = [450, 300, 150]
    
    df_vol = df_book_processed.groupby('time_id')['log_return'].agg(realized_volatility).reset_index()
    df_vol.columns = ['time_id', 'volatility_realized']
    for window in windows:
        df_book_window = df_book_processed[df_book_processed['seconds_in_bucket'] >= 600 - window]
        df_vol_window = df_book_window.groupby('time_id')['log_return'].agg(realized_volatility).reset_index()
        df_vol_window.columns = ['time_id', f'volatility_realized_{window}s']
        df_vol = pd.merge(df_vol, df_vol_window, on='time_id', how='left')

    def create_agg_features(df, group_col, value_col, windows):
        df_agg = df.groupby(group_col)[value_col].agg(['mean', 'std', 'sum']).reset_index()
        df_agg.columns = [group_col, f'{value_col}_mean', f'{value_col}_std', f'{value_col}_sum']
        for window in windows:
            df_window = df[df['seconds_in_bucket'] >= 600 - window]
            df_window_agg = df_window.groupby(group_col)[value_col].agg(['mean', 'std', 'sum']).reset_index()
            df_window_agg.columns = [group_col, f'{value_col}_mean_{window}s', f'{value_col}_std_{window}s', f'{value_col}_sum_{window}s']
            df_agg = pd.merge(df_agg, df_window_agg, on=group_col, how='left')
        return df_agg

    df_obi_agg = create_agg_features(df_book_processed, 'time_id', 'obi', windows)
    df_spread_agg = create_agg_features(df_book_processed, 'time_id', 'spread', windows)
    df_trade_agg = create_agg_features(df_trade, 'time_id', 'size', windows)
    
    df_features = pd.merge(df_vol, df_obi_agg, on='time_id', how='left')
    df_features = pd.merge(df_features, df_spread_agg, on='time_id', how='left')
    df_features = pd.merge(df_features, df_trade_agg, on='time_id', how='left')
    
    df_features['stock_id'] = stock_id
    return df_features

train_stock_ids = train_df['stock_id'].unique()
all_features_list = []
for stock_id in tqdm(train_stock_ids, desc="Generating Fixed Advanced Features"):
    stock_features = generate_all_features_fixed(stock_id, data_dir)
    all_features_list.append(stock_features)

features_df_advanced = pd.concat(all_features_list, ignore_index=True)
print("\n Generated high-level features (partial)")
print(features_df_advanced.head())


generate_all_features_universal = generate_all_features_fixed


# build train/test features by stock_id
from tqdm import tqdm
import numpy as np
import pandas as pd

def build_all_features(split: str, sids: np.ndarray) -> pd.DataFrame:
    parts = []
    for sid in tqdm(sids, desc=f"features-{split}"):
        feats_sid = generate_all_features_universal(int(sid), data_dir=data_dir, data_type=split)
        parts.append(feats_sid)
    if parts:
        feats = pd.concat(parts, axis=0, ignore_index=True)
   
        for c in ("stock_id", "time_id"):
            if c in feats.columns:
                feats[c] = feats[c].astype("int32", copy=False)
        feats = feats.sort_values(["stock_id","time_id"], kind="mergesort").reset_index(drop=True)
        return feats
    else:
        return pd.DataFrame(columns=["stock_id","time_id"])

train_sids = train_df["stock_id"].unique()
test_sids  = test_df["stock_id"].unique()

print("Using pre-computed features from Cell 3 for the training set.")
train_feats = features_df_advanced 

test_feats  = build_all_features("test", test_sids)

print("train_feats:", train_feats.shape, " test_feats:", test_feats.shape)




import lightgbm as lgb
import joblib
from pathlib import Path
import numpy as np
import pandas as pd


def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

train_advanced_df = pd.merge(train_df, features_df_advanced, on=['stock_id', 'time_id'], how='left')
train_advanced_df = train_advanced_df.fillna(0)

features_advanced = [col for col in features_df_advanced.columns if col != 'time_id']

train_advanced_df['stock_id'] = train_advanced_df['stock_id'].astype('category')

X_adv = train_advanced_df[features_advanced]
y_adv = train_advanced_df['target']
groups_adv = train_advanced_df['time_id']

print(f"\nNumber of high-level features used: {len(features_advanced)}")
print("Skipping model training as we will load pre-trained models.")

print("\n Loading Pre-trained Models ")

MODEL_DIR = Path('/kaggle/input/lgbrmspe5') 

# 您的5个模型文件名列表
model_files = [
    'lgb_rmspe_eval_fold1.pkl',
    'lgb_rmspe_eval_fold2.pkl',
    'lgb_rmspe_eval_fold3.pkl',
    'lgb_rmspe_eval_fold4.pkl',
    'lgb_rmspe_eval_fold5.pkl'
]

models_adv = []
print("Start loading the 5-fold cross validation model")

all_models_found = True
for file_name in model_files:
    model_path = MODEL_DIR / file_name
    
    if model_path.is_file():
        print(f"Loading the model:{model_path}")
        model = joblib.load(model_path)
        models_adv.append(model)
    else:
        print(f"Error: Model file not found at {model_path}")
        all_models_found = False


if all_models_found and len(models_adv) == 5:
    print(f"\nSuccessfully loaded {len(models_adv)} models")
else:
    raise FileNotFoundError(f" Please check the path'{MODEL_DIR}' ")


import numpy as np
import pandas as pd
from pathlib import Path


models = models_adv
m0 = models[0]
try:
    feat_from_model = list(m0.booster_.feature_name())
except Exception:
    feat_from_model = list(getattr(m0, "feature_name_", []))
if feat_from_model:
    features_advanced = feat_from_model

print(f"Using {len(models)} models; num features = {len(features_advanced)}")

test_merged = pd.merge(
    test_df[['row_id','stock_id','time_id']],
    test_feats, on=['stock_id','time_id'], how='left'
).fillna(0)

X_test = test_merged.reindex(columns=features_advanced).fillna(0)
if 'stock_id' in X_test.columns:
    X_test['stock_id'] = X_test['stock_id'].astype('category')

pred = np.zeros(len(X_test), dtype=float)
for m in models:
    num_it = getattr(m, "best_iteration_", None)
    pred += m.predict(X_test, num_iteration=num_it) / len(models)

sub_path = Path("/kaggle/working") / "submission.csv"
pd.DataFrame({"row_id": test_merged["row_id"].astype(str), "target": pred}).to_csv(sub_path, index=False)
print("Saved:", sub_path)


