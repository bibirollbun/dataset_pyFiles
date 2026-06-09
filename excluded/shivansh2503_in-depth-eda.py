import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


df.head()


df['Fertilizer Name'].value_counts()




