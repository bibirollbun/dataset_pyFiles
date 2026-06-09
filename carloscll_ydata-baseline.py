%%capture
!pip install ydata_profiling


import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
profile = ProfileReport(df, title="Profiling Report")


profile

