train_csv = "/kaggle/input/playground-series-s4e12/train.csv"
test_csv = "/kaggle/input/playground-series-s4e12/test.csv"


import pandas as pd
pd.set_option('display.max_columns', 500)
train_df = pd.read_csv(train_csv, index_col="id")
test_df = pd.read_csv(test_csv, index_col="id")


pd

