# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

def composite_score(y_true_item, y_pred_item, y_true_qty, y_pred_qty):
    acc = accuracy_score(y_true_item, y_pred_item)
    f1  = f1_score(y_true_item, y_pred_item, average='macro')
    if np.all(y_true_qty == y_true_qty[0]):
        reg_score = 1.0
    else:
        mae = mean_absolute_error(y_true_qty, y_pred_qty)
        rng = float(np.max(y_true_qty) - np.min(y_true_qty))
        norm_mae = mae / rng if rng > 0 else 0.0
        norm_mae = np.clip(norm_mae, 0, 1)
        reg_score = 1 - norm_mae
    return 0.25*acc + 0.25*f1 + 0.5*reg_score



# data_utils.py
import pandas as pd
import numpy as np
from typing import Tuple

SIZE_BINS = [-np.inf, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, np.inf]
SIZE_LABELS = ["XS","S","M","L","XL","XXL","XXXL"]

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def safe_to_numeric(df: pd.DataFrame, col: str):
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def parse_dates(df: pd.DataFrame):
    # Try to parse common date columns if present
    for col in ["invoiceDate","CONSTRUCTION_START_DATE","SUBSTANTIAL_COMPLETION_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def bin_size(col_val):
    try:
        v = float(col_val)
    except Exception:
        return np.nan
    return pd.cut([v], bins=SIZE_BINS, labels=SIZE_LABELS)[0]

def add_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for df in (train, test):
        df["SIZE_BIN"] = df.get("SIZE_BUILDINGSIZE", np.nan).apply(bin_size)
        parse_dates(df)
        # Basic derived features
        if "CONSTRUCTION_START_DATE" in df.columns and "SUBSTANTIAL_COMPLETION_DATE" in df.columns:
            df["project_duration_days"] = (df["SUBSTANTIAL_COMPLETION_DATE"] - df["CONSTRUCTION_START_DATE"]).dt.days
            df["start_month"] = df["CONSTRUCTION_START_DATE"].dt.month
    # Ensure numeric columns for training safe conversion
    train = safe_to_numeric(train, "MasterItemNo")
    train = safe_to_numeric(train, "QtyShipped")
    return train, test



# baseline_model.py
import pandas as pd
import numpy as np
from typing import List, Tuple

def build_agg_maps(train: pd.DataFrame, group_orders: List[List[str]]):
    agg_maps = []
    for cols in group_orders:
        # include dropna=False to keep NaNs as keys
        grp = train.groupby(cols, dropna=False).agg(
            MasterItemNo_mode=("MasterItemNo", lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else np.nan),
            Qty_med=("QtyShipped", lambda s: float(s.dropna().median()) if not s.dropna().empty else np.nan)
        ).reset_index()
        agg_maps.append((cols, grp))
    return agg_maps

def predict_hierarchical(test: pd.DataFrame, agg_maps: List[Tuple[List[str], pd.DataFrame]], train: pd.DataFrame):
    pred = pd.DataFrame({"id": test["id"], "MasterItemNo": np.nan, "QtyShipped": np.nan})
    work = test.copy()

    for cols, mp in agg_maps:
        merged = work.merge(mp, on=cols, how="left", suffixes=("","_agg"))
        need_item = pred["MasterItemNo"].isna()
        if "MasterItemNo_mode" in merged.columns:
            pred.loc[need_item, "MasterItemNo"] = merged.loc[need_item, "MasterItemNo_mode"].values
        need_qty = pred["QtyShipped"].isna()
        if "Qty_med" in merged.columns:
            pred.loc[need_qty, "QtyShipped"] = merged.loc[need_qty, "Qty_med"].values

    # Fallbacks
    global_item = train["MasterItemNo"].dropna().mode().iloc[0] if not train["MasterItemNo"].dropna().mode().empty else 0
    global_qty = float(train["QtyShipped"].dropna().median()) if not train["QtyShipped"].dropna().empty else 1.0

    pred["MasterItemNo"] = pred["MasterItemNo"].fillna(global_item).astype(int)
    pred["QtyShipped"] = pred["QtyShipped"].fillna(global_qty)

    # Round quantities: if training quantities are integer-like, round to int
    is_integer_like = (train["QtyShipped"].dropna() % 1 == 0).all() if not train["QtyShipped"].dropna().empty else True
    if is_integer_like:
        pred["QtyShipped"] = pred["QtyShipped"].round().astype(int)
    else:
        pred["QtyShipped"] = pred["QtyShipped"].round(2)

    # ensure non-negative
    pred["QtyShipped"] = pred["QtyShipped"].clip(lower=0)

    return pred[["id","MasterItemNo","QtyShipped"]]



# metric.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

def composite_score(y_true_item, y_pred_item, y_true_qty, y_pred_qty):
    acc = accuracy_score(y_true_item, y_pred_item)
    # If there is only one class present, f1_score with macro might warn; handle gracefully
    try:
        f1  = f1_score(y_true_item, y_pred_item, average='macro', zero_division=0)
    except Exception:
        f1 = 0.0
    if np.all(y_true_qty == y_true_qty[0]):
        reg_score = 1.0
    else:
        mae = mean_absolute_error(y_true_qty, y_pred_qty)
        rng = float(np.max(y_true_qty) - np.min(y_true_qty))
        norm_mae = mae / rng if rng > 0 else 0.0
        norm_mae = np.clip(norm_mae, 0, 1)
        reg_score = 1 - norm_mae
    return 0.25*acc + 0.25*f1 + 0.5*reg_score, {"acc":acc, "f1":f1, "reg_score":reg_score, "mae":mae if 'mae' in locals() else 0.0}



import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Data Utils
# -----------------------------
SIZE_BINS = [-np.inf, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, np.inf]
SIZE_LABELS = ["XS","S","M","L","XL","XXL","XXXL"]

def bin_size(val):
    try:
        v = float(val)
    except Exception:
        return np.nan
    return pd.cut([v], bins=SIZE_BINS, labels=SIZE_LABELS)[0]

def add_features(train, test):
    for df in (train, test):
        if "SIZE_BUILDINGSIZE" in df.columns:
            df["SIZE_BIN"] = df["SIZE_BUILDINGSIZE"].apply(bin_size)
    train["MasterItemNo"] = pd.to_numeric(train["MasterItemNo"], errors="coerce")
    train["QtyShipped"]   = pd.to_numeric(train["QtyShipped"], errors="coerce")
    return train, test

# -----------------------------
# Baseline Model
# -----------------------------
def build_agg_maps(train, group_orders):
    agg_maps = []
    for cols in group_orders:
        grp = train.groupby(cols, dropna=False).agg(
            MasterItemNo_mode=("MasterItemNo", lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else np.nan),
            Qty_med=("QtyShipped", lambda s: float(s.dropna().median()) if not s.dropna().empty else np.nan)
        ).reset_index()
        agg_maps.append((cols, grp))
    return agg_maps

def predict_hierarchical(test, agg_maps, train):
    pred = pd.DataFrame({"id": test["id"], "MasterItemNo": np.nan, "QtyShipped": np.nan})
    work = test.copy()
    for cols, mp in agg_maps:
        merged = work.merge(mp, on=cols, how="left")
        need_item = pred["MasterItemNo"].isna()
        pred.loc[need_item, "MasterItemNo"] = merged.loc[need_item, "MasterItemNo_mode"].values
        need_qty = pred["QtyShipped"].isna()
        pred.loc[need_qty, "QtyShipped"] = merged.loc[need_qty, "Qty_med"].values

    global_item = train["MasterItemNo"].dropna().mode().iloc[0] if not train["MasterItemNo"].dropna().mode().empty else 0
    global_qty = float(train["QtyShipped"].dropna().median()) if not train["QtyShipped"].dropna().empty else 1.0

    pred["MasterItemNo"] = pred["MasterItemNo"].fillna(global_item).astype(int)
    pred["QtyShipped"] = pred["QtyShipped"].fillna(global_qty)
    is_integer_like = (train["QtyShipped"].dropna() % 1 == 0).all()
    if is_integer_like:
        pred["QtyShipped"] = pred["QtyShipped"].round().astype(int)
    else:
        pred["QtyShipped"] = pred["QtyShipped"].round(2)
    pred["QtyShipped"] = pred["QtyShipped"].clip(lower=0)
    return pred[["id","MasterItemNo","QtyShipped"]]

# -----------------------------
# Run Inference
# -----------------------------
base = Path("/mnt/data/ctai-ctd-hackathon")
train = pd.read_csv(base/"/kaggle/input/ctai-ctd-hackathon/train.csv")
test = pd.read_csv(base/"/kaggle/input/ctai-ctd-hackathon/test.csv")

train, test = add_features(train, test)

group_orders = [
    ["PROJECT_TYPE","CORE_MARKET","STATE","SIZE_BIN"],
    ["PROJECT_TYPE","CORE_MARKET","STATE"],
    ["PROJECT_TYPE","CORE_MARKET"],
    ["PROJECT_TYPE"],
    ["CORE_MARKET","STATE"],
    ["CORE_MARKET"],
    ["STATE"],
]

agg_maps = build_agg_maps(train, group_orders)
submission = predict_hierarchical(test, agg_maps, train)

#submission.to_csv(base/"submission_notebook.csv", index=False)
submission





