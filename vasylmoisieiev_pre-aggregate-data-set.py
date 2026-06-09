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


!pip install dask


!python -m pip install "dask[distributed]" --upgrade 


import os
import dask.dataframe as dd
import pandas as pd
import os, shutil
from dask.distributed import Client


client = Client(n_workers=8, threads_per_worker=2, memory_limit="32GB")
print(client)

ID_COL = "customer_ID"

TRAIN_PATH  = "/kaggle/input/amex-default-prediction/train_data.csv"
TEST_PATH   = "/kaggle/input/amex-default-prediction/test_data.csv"
LABELS_PATH = "/kaggle/input/amex-default-prediction/train_labels.csv"

TRAIN_OUT       = "/kaggle/working/train_agg.parquet"
TEST_OUT        = "/kaggle/working/test_agg.parquet"
FINAL_TRAIN_OUT = "/kaggle/working/train_final.parquet"

ddf_train = dd.read_csv(TRAIN_PATH, blocksize="2GB")
ddf_test  = dd.read_csv(TEST_PATH, blocksize="2GB")

num_cols = ddf_train.select_dtypes(include=["number"]).columns.tolist()
print("Numeric cols:", len(num_cols))

aggs = ["mean", "std", "min", "max", "last"]

def aggregate(ddf, num_cols, id_col=ID_COL, split_out=128):
    agg_ddf = (
        ddf.groupby(id_col)[num_cols]
        .agg(aggs, split_out=split_out)
        .reset_index()
    )
    agg_ddf.columns = [
        col if col == id_col else f"{col[0]}_{col[1]}"
        for col in agg_ddf.columns.values
    ]
    return agg_ddf

train_agg_ddf = aggregate(ddf_train, num_cols, split_out=256)
test_agg_ddf  = aggregate(ddf_test, num_cols, split_out=256)

def save_parquet(ddf, path):
    if os.path.exists(path):
        shutil.rmtree(path)
    ddf.to_parquet(
        path,
        engine="pyarrow",
        compression="zstd",  
        write_index=False
    )
    print(f"✔ Saved: {path}")

save_parquet(train_agg_ddf, TRAIN_OUT)
save_parquet(test_agg_ddf, TEST_OUT)

train_agg = dd.read_parquet(TRAIN_OUT).compute()

id_col_in_parquet = [c for c in train_agg.columns if "customer" in c.lower()][0]
print("ID column in parquet:", id_col_in_parquet)

labels = pd.read_csv(LABELS_PATH)

train_final = train_agg.merge(labels, left_on=id_col_in_parquet, right_on=ID_COL, how="inner")

float_cols = train_final.select_dtypes(include=["float64"]).columns
train_final[float_cols] = train_final[float_cols].astype("float32")

train_final.to_parquet(
    FINAL_TRAIN_OUT,
    engine="pyarrow",
    compression="zstd",
    index=False
)

print("Train final shape:", train_final.shape)
print(f"✔ Final train saved: {FINAL_TRAIN_OUT}")
print(f"✔ Test agg saved: {TEST_OUT}")


print(train_agg.columns[:20])   
print(train_agg.head())


print(os.path.getsize(TRAIN_OUT) / 1e9, "GB")
print(os.path.getsize(FINAL_TRAIN_OUT) / 1e9, "GB")


train = pd.read_parquet(TRAIN_OUT)
test  = pd.read_parquet(FINAL_TRAIN_OUT)

print(train.shape)
print(test.shape)


print(os.listdir("."))  
print(os.listdir("/kaggle/working"))  

