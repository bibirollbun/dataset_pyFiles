import pandas as pd
from eda_utils_script import quick_eda

df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
quick_eda(df)


