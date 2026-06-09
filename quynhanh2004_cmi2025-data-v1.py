


!pip install -q polars==1.29.0

import numpy as np, pandas as pd, polars as pl
from gc import collect
from colorama import Fore, Back, Style
from tqdm.notebook import tqdm

def PrintColor(text: str, color = Fore.BLUE, style = Style.BRIGHT) :
    print( color + style + text + Style.RESET_ALL)


%%time 

train      = \
(
    pl.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv").
    select(pl.all().shrink_dtype())
)
train_demo = pl.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

PrintColor(f"---> Columns in device data-  train\n")
with np.printoptions(linewidth = 100, threshold = 10000) :
    print(np.array(train.collect_schema().names()))

train.write_parquet("train.parquet", partition_by = "subject")

print(
    f"\n\n---> Estimated memory for the training data = {train.estimated_size() / 10**6} Mb\n"
)

train_demo.write_csv(f"train_demo.csv")

test = \
(
    pl.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv").
    select(pl.all().shrink_dtype())
)
test_demo = pl.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")

test.write_parquet("test.parquet", partition_by = "subject")
test_demo.write_csv(f"test_demo.csv")

collect()


nulls_ = train.null_count() / train.height
nulls_ = nulls_.to_pandas().transpose().squeeze()

PrintColor(f"---> Top 25 columns by null values")
display(
    nulls_.sort_values(ascending = False).loc[nulls_ > 0].
    head(25)
)

